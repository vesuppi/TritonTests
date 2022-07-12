import sys
import torch
import triton 
import triton.language as tl
from utils import *
from torchinductor.triton_ops.batched_matmul import bmm_out
from configs import basic_configs


@triton.jit
def _kernel1(a_cols, a_vals, b_vals, c_vals, 
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
            nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
            ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    # pid = tl.program_id(0)
    # bid = tl.program_id(1)
    # m = pid // nBN
    # n = pid % nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # b_cols = n * BN + tl.arange(0, BN)
    # b_ptrs = b_vals + tl.arange(0, BK)[:, None] * N + b_cols[None, :]


    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    k_start = tl.load(a_cols + 2*m)
    k_end = tl.load(a_cols + 2*m+1)
    a_ptrs = a_ptrs+a_block_size * k_start
    b_ptrs = b_ptrs+b_block_size * nBN * k_start

    c = tl.zeros((BM, BN), dtype=tl.float32)

    # for k in range(nBK):
    #     a = tl.load(a_ptrs)
    #     b = tl.load(b_ptrs)
    #     c += tl.dot(a, b)
    #     a_ptrs += a_block_size
    #     b_ptrs += b_block_size * nBN

    for _ in range(k_start, k_end):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN

    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def bmm1(B, M, K, N, BM, BK, BN, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM = cdiv(M, BM)
    nBN = cdiv(N, BN)
    nBK = cdiv(K, BK)
    grid = (nBM, nBN, B)
    binary = _kernel1[grid](a_cols, a_vals, b_vals, c,
                            M, K, N, 
                            BM, BK, BN, nBM, nBK, nBN, 
                            num_warps=num_warps, num_stages=num_stages
                            )
    #print(binary.asm['ptx'])
    return c

# def bmm1(a: MCSR, b: MCSR, c, num_warps=4, num_stages=3):
#     B, nBM, nBK, BM, BK = a.vals.shape
#     B, nBK, nBN, BK, BN = b.vals.shape
#     # TODO: this does not work when M does not divide BM
#     # Or maybe it works because C will also need to be padded
#     M = nBM * BM 
#     N = nBN * BN

#     grid = (nBM * nBN, B)
#     #print(grid)
    
#     binary = _kernel1[grid](a.rowptrs, a.cols, a.vals, b.vals, c[1],
#                                     BM, BK, BN, nBM, nBK, nBN, 
#                                     num_warps=num_warps, num_stages=num_stages
#                                     )
#     #print(binary.asm['ptx'])
#     return c

