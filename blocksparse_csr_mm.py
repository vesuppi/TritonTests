import sys
import torch
print('imported torch')
import triton 
import triton.language as tl
import utils
from utils import *


@triton.jit
def _kernel_mcsr_mm(a_rowptrs, a_cols, a_vals, b_vals, c_vals, 
                                BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
                                nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
                                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # a_rowptrs_m = a_rowptrs + m
    k_start = tl.load(a_rowptrs+m)
    k_end = tl.load(a_rowptrs+m+1)
    c = tl.zeros((BM, BN), dtype=tl.float32)

    # for k in range(nBK):
    #     a = tl.load(a_ptrs)
    #     b = tl.load(b_ptrs)
    #     c += tl.dot(a, b)

    #     a_ptrs += a_block_size
    #     b_ptrs += b_block_size * nBN

    
    for kp in range(k_start, k_end):
        k = tl.load(a_cols+kp)
        a = tl.load(a_ptrs+a_block_size*k)
        b = tl.load(b_ptrs+b_block_size * nBN*k)
        c += tl.dot(a, b)
        
    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(c_ptrs, c)


def mcsr_mm_inner(a_rowptrs, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM, nBK, BM, BK = a_vals.shape
    nBK, nBN, BK, BN = b_vals.shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN

    grid = (nBM, nBN)
    binary = _kernel_mcsr_mm[grid](a_rowptrs, a_cols, a_vals, b_vals, c,
                                    BM, BK, BN, nBM, nBK, nBN, 
                                    num_warps=num_warps, num_stages=num_stages
                                    )
    #print(binary.asm['ptx'])
    return c

def mcsr_mm(a: MCSR, b: MCSR, c, num_warps=4, num_stages=3):
    nBM, nBK, BM, BK = a.vals.shape
    nBK, nBN, BK, BN = b.vals.shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN

    grid = (nBM, nBN)
    #print(grid)
    
    binary = _kernel_mcsr_mm[grid](a.rowptrs, a.cols, a.vals, b.vals, c[1],
                                    BM, BK, BN, nBM, nBK, nBN, 
                                    num_warps=num_warps, num_stages=num_stages
                                    )
    #print(binary.asm['ptx'])
    return c


def verify_run():
    M = 32
    K = M
    N = M

    BM = 16
    BK = BM
    BN = BM
    a = gen_lower_triangular_mcsr_matrix(M, K, BM, BK)
    b = gen_random_mcsr_matrix(K, N, BK, BN, density=1)
    c = gen_empty_matrix_dense_blocks(M, N, BM, BN)
    a_ref = from_block_format(a.vals)
    b_ref = from_block_format(b.vals)
    c_ref = torch.mm(a_ref, b_ref)
    c = mcsr_mm(a, b, c)
    print('verify passes:', torch.allclose(c_ref, from_block_format(c[1])))


def test_random():
    M = 1024
    K = 1024
    N = M
    
    BMs = [32, 64, 128, 256]
    BKs = [32, 64, 128, 256]
    BNs = [32, 64, 128, 256]
    #stages = [1,2,3,4,5]
    #warps = [1,2,4,8]
    stages = [2,3,4,5]
    warps = [1,2,4]

    TEST_RUN = False

    if TEST_RUN:
        BMs, BKs, BNs = [64], [64], [64]
        stages, warps = [1,2,3,4,5], [1,2,4]

    
    for BM in BMs:
        for BK in BKs:
            for BN in BNs:
                #if BM * K != BK * M:
                    #continue
                a = gen_random_mcsr_matrix(M, K, BM, BK, density=1)
                #a = gen_lower_triangular_mcsr_matrix(M, K, BM, BK)
                #a = gen_lower_half_mcsr_matrix(M, K, BM, BK)
                b = gen_random_mcsr_matrix(K, N, BK, BN, density=1)
                c = gen_empty_matrix_dense_blocks(M, N, BM, BN)
                a_ref = from_block_format(a.vals)
                b_ref = from_block_format(b.vals)
                c_ref = torch.empty(M, N, dtype=torch.float16, device='cuda')
                ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a_ref, b_ref, out=c_ref))
                print(f'torch mm: {ms:.4f}')

                times = []
                ms = torch.inf
                try:
                    for num_stages in stages:
                        for num_warps in warps:
                            ms, _, _ = triton.testing.do_bench(lambda: mcsr_mm(a, b, c, num_warps, num_stages), rep=50)
                            times.append((ms, BM, BK, BN, num_stages, num_warps))
                except Exception as e:
                    print('run triton failed')
                    continue
                print('verify passes:', torch.allclose(c_ref, from_block_format(c[1])))
                times.sort(key=lambda x: x[0])
                print(times[0])
                print(f'blocksparse mm: {times[0][0]:.4f} ({BM} x {BK} x {BN})')

    
def test_lower_triangular():
    M = 3072
    K = M 
    N = M

    dtype = torch.float16
    a = torch.randn([M, K], dtype=dtype, device='cuda')
    a[M//2:, :] = 0
    #a[:, K//2:] = 0
    #a = torch.tril(a)
    b = torch.randn([K, N], dtype=dtype, device='cuda')
    c_ref = torch.empty(M, N, dtype=dtype, device='cuda')
    ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a, b, out=c_ref))
    print(f'torch mm: {ms:.4f}')

    BMs = [32, 64, 128, 256]
    BKs = [16, 32, 64, 128, 256]
    BNs = [32, 64, 128]
    #stages = [1,2,3,4,5]
    #warps = [1,2,4,8]
    stages = [2,3,4,5]
    warps = [1,2,4]

    TEST_RUN = False

    if TEST_RUN:
        s = 16
        BMs, BKs, BNs = [s], [s], [s]
        stages, warps = [1,2,3,4,5], [1,2,4]

    best_time = torch.inf
    
    for BM in BMs:
        for BK in BKs:
            for BN in BNs:
                if BM * K != BK * M:
                    continue
                a_block, a_mask = utils.to_block_format_with_mask(a, BM, BK)
                #print(a_mask)
                a_mask_rowptrs, a_mask_cols = utils.to_csr_ptrs(a_mask)
                b_block = utils.to_block_format(b, BK, BN)
                #print(a_mask_rowptrs, a_mask_cols)
                c = gen_empty_matrix_dense_blocks(M, N, BM, BN)

                
                times = []
                ms = torch.inf
                try:
                    for num_stages in stages:
                        for num_warps in warps:
                            ms, _, _ = triton.testing.do_bench(lambda: mcsr_mm_inner(a_mask_rowptrs, a_mask_cols, a_block, b_block, c[1], num_warps, num_stages), rep=50)
                            
                            times.append((ms, BM, BK, BN, num_stages, num_warps))
                except Exception as e:
                    print('run triton failed')
                    continue
                verified = torch.allclose(c_ref, utils.from_block_format(c[1]))
                print('verify passes:', verified)
                if verified:
                    times.sort(key=lambda x: x[0])
                    print(f'info: blocksparse mm: {times[0][0]:.4f} ({BM} x {BK} x {BN})')
                    if times[0][0] < best_time:
                        best_time = times[0][0]

    print(f'blocksparse mm: {best_time:.5f}')

    
#test_random()
test_lower_triangular()