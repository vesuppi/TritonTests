import sys
import torch
import triton 
import triton.language as tl
from utils import *


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)



def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y



@triton.jit
def _rowwise_softmax_kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, 
                BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    block_ptrs = x_ptr + m * BM * N + tl.arange(0, BN)
    row = tl.load(block_ptrs)
    max = tl.max(row, axis=0)
    normalized_row = row - max

    t0 = tl.exp(normalized_row)
    t1 = tl.sum(t0, axis=0)
    y = t0 / t1
    y_ptrs = y_ptr + m * BM * N + tl.arange(0, BN)
    tl.store(y_ptrs, y)


def rowwise_softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    BM = 1
    BN = triton.next_power_of_2(N)
    #softmax_kernel[grid](y, x, N, N, N, N)
    _rowwise_softmax_kernel[grid](x, y, M, N, BM, BN)
    return y