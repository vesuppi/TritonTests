import sys
import torch
import triton 
import triton.language as tl
from utils import *


class NewFormat:
    def __init__(self, mask) -> None:        
        self.rowptrs, self.cols = to_contiguous_nz_format_simple(mask)
       


@triton.jit
def _exp_1d_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                # tile sizes, BM needs to divide TM etc
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    m = tl.program_id(0)
    block_size = BM * BN

    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*m)
    col_end = tl.load(x_cols + 2*m+1)

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += m * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass
    offsets += tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :] 
    x_offsets = x_data + offsets
    y_offsets = y_data + offsets

    for _ in range(col_start, col_end):
        ## Format specific: how to get `k` would depend on the format
        block = tl.load(x_offsets)
        
        ## Kernel specific
        block = tl.exp(block)
        
        ## Format specific: how to get `k` would depend on the format
        tl.store(y_offsets, block)
        x_offsets += block_size
        y_offsets += block_size


def exp_1d(x_mask: NewFormat, x_data):
    '''
    Launch a 1D grid to do the computation (blocking rows only).

    2x slower than the 2D blocking kernel for dense matrices of shape > 4096 x 4096.
    '''
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    y_data = torch.empty_like(x_data)
    ## Grid size: not blocking the columns. Tow few thread blocks?
    grid = (m, B)    
    _exp_1d_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, BM, BN, True
    )
    # Same mask is used for y
    return (x_mask, y_data)



@triton.jit
def _exp_2d_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                # tile sizes, BM needs to divide TM etc
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*m)
    col_end = tl.load(x_cols + 2*m+1)
    if (n >= col_end) | (n < col_start):
        return

    block_size = BM * BN

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += m * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass
    offsets += tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :] 
    offsets += n * block_size
    x_offsets = x_data + offsets
    y_offsets = y_data + offsets

    ## Format specific: how to get `k` would depend on the format
    block = tl.load(x_offsets)
    
    ## Kernel specific
    block = tl.exp(block)
    
    ## Format specific: how to get `k` would depend on the format
    tl.store(y_offsets, block)


def exp_2d(x_mask: NewFormat, x_data):
    '''
    Launch a 2D grid to do the computation.

    Achieved comparable performance with torch.exp for dense matrices of shape > 4096 x 4096.
    '''
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    y_data = torch.empty_like(x_data)
    ## Grid size: not blocking the columns. Tow few thread blocks?
    grid = (m, n, B)    
    _exp_2d_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, BM, BN, True
    )
    # Same mask is used for y
    return (x_mask, y_data)







@triton.jit
def _sum_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                # tile sizes, BM needs to divide TM etc
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    m = tl.program_id(0)
    block_size = BM * BN

    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*m)
    col_end = tl.load(x_cols + 2*m+1)

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += m * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass
    offsets += tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :] 
    x_offsets = x_data + offsets
    y_offsets = y_data + offsets

    c = tl.zeros((BM, 1))
    for _ in range(col_start, col_end):
        ## Format specific: how to get `k` would depend on the format
        block = tl.load(x_offsets)

        ## Kernel
        c += tl.sum(block, axis=1)

        ## Format specific: how to get `k` would depend on the format
        x_offsets += block_size

    tl.store(y_offsets, c)


def sum(x_mask: NewFormat, x_data):
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    #y_data = torch.zeros_like(x_data)
    ## Grid size: not blocking the columns
    grid = (m, B)    
    _sum_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, BM, BN, True
    )
    # Same mask is used for y
    return (x_mask, y_data)






# @triton.jit
# def _rowwise_softmax_kernel(x_rowptrs, x_cols, x_data, y_data, M: tl.constexpr, N: tl.constexpr, 
#                 BM: tl.constexpr, BN: tl.constexpr, use_dense_data: tl.constexpr,
#                 TM: tl.constexpr, TN: tl.constexpr):
#     m = tl.program_id(0)
#     k_start = tl.load(x_cols + 2*m)
#     k_end = tl.load(x_cols + 2*m+1)

#     block_ptrs = x_data + tl.arange(0, BN)

#     if use_dense_data:
#         block_ptrs += m * BM * N

#     row = tl.load(block_ptrs)
#     max = tl.max(row, axis=0)
#     normalized_row = row - max

#     t0 = tl.exp(normalized_row)
#     t1 = tl.sum(t0, axis=0)
#     y = t0 / t1
#     y_ptrs = y_ptr + m * BM * N + tl.arange(0, BN)
#     tl.store(y_ptrs, y)


# def rowwise_softmax(x_mask: FastTrackMask, x_data, M, N):
#     y_data = torch.empty_like(x_data)
#     grid = (M,)
#     BM = 1
#     BN = triton.next_power_of_2(N)
#     #softmax_kernel[grid](y, x, N, N, N, N)
#     _rowwise_softmax_kernel[grid](x_mask.rowptrs, x_mask.cols, y_data, M, N, BM, BN)
#     return (x_mask, y_data)