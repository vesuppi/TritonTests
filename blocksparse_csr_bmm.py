import sys
import torch
print('imported torch')
import triton 
import triton.language as tl
from utils import *
from torchinductor.triton_ops.batched_matmul import bmm_out
import argparse


@triton.jit
def _kernel_mcsr_bmm(a_cols, a_vals, b_vals, c_vals, 
                                BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
                                nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
                                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    M: tl.constexpr = BM * nBM
    K: tl.constexpr = BK * nBK
    N: tl.constexpr = BN * nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    # k_start = tl.load(a_cols + 2*m)
    # k_end = tl.load(a_cols + 2*m+1)
    # a_ptrs = a_ptrs+a_block_size * k_start
    # b_ptrs = b_ptrs+b_block_size * nBN * k_start

    c = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(nBK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN

    # for _ in range(k_start, k_end):
    #     a = tl.load(a_ptrs)
    #     b = tl.load(b_ptrs)
    #     c += tl.dot(a, b)
    #     a_ptrs += a_block_size
    #     b_ptrs += b_block_size * nBN

    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def mcsr_bmm_inner(a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    B, nBM, nBK, BM, BK = a_vals.shape
    B, nBK, nBN, BK, BN = b_vals.shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN

    grid = (nBM, nBN, B)
    binary = _kernel_mcsr_bmm[grid](a_cols, a_vals, b_vals, c,
                                    BM, BK, BN, nBM, nBK, nBN, 
                                    num_warps=num_warps, num_stages=num_stages
                                    )
    #print(binary.asm['ptx'])
    return c

def mcsr_mm(a: MCSR, b: MCSR, c, num_warps=4, num_stages=3):
    B, nBM, nBK, BM, BK = a.vals.shape
    B, nBK, nBN, BK, BN = b.vals.shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN

    grid = (nBM, nBN, B)
    #print(grid)
    
    binary = _kernel_mcsr_bmm[grid](a.rowptrs, a.cols, a.vals, b.vals, c[1],
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


    
def test_lower_triangular(B, M, K, N):
    # B = 10
    # M = 1024
    # K = M 
    # N = M

    TEST_RUN = False
    if TEST_RUN:
        B = 2
        M = 8
        K = M
        N = M

    dtype = torch.float16
    a = torch.randn([B, M, K], dtype=dtype, device='cuda')
    #a[M//2:, :] = 0
    #a[:, K//2:] = 0
    #a = torch.tril(a)
    b = torch.randn([B, K, N], dtype=dtype, device='cuda')
    c_ref = torch.empty([B, M, N], dtype=dtype, device='cuda')
    torch_ms, _, _ = triton.testing.do_bench(lambda: torch.bmm(a, b, out=c_ref))
    print(f'info: torch bmm: {torch_ms:.4f}')

    triton_c_ref = torch.empty([B, M, N], dtype=dtype, device='cuda')
    #triton_ms, _, _ = triton.testing.do_bench(lambda: bmm_out(a, b, triton_c_ref))
    triton_ms = 0
    print(f'info: triton bmm: {triton_ms:.4f}')
    print(torch.allclose(c_ref, triton_c_ref, atol=0.1, rtol=0.01))

    #sys.exit(1)

    #BMs = [32, 64, 128, 256]
    #BKs = [32, 64, 128, 256]
    BMs = [32, 64, 128]
    BKs = [32, 64, 128]
    BNs = [32, 64, 128]
    #stages = [1,2,3,4,5]
    #warps = [1,2,4,8]
    stages = [2,3,4,5]
    warps = [1,2,4]
    

    if TEST_RUN:
        s = 4
        BMs, BKs, BNs = [s], [s], [s]
        stages, warps = [2,3,4,5], [1,2,4]

    best_time = torch.inf
    print(f'info: shapes: {B} x {M} x {K} x {N}')
    for BM in BMs:
        for BK in BKs:
            for BN in BNs:
                if BM > M or BK > K or BN > N:
                    continue
                
                if BM * K != BK * M:
                    continue

                
                print(f'info: blocks: {BM} x {BK} x {BN}')
                a_block, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BK)
                a_mask_cols = to_contiguous_nz_format_simple(a_mask)
                
                #print(a_mask_cols)
                #sys.exit(1)
                b_block, b_mask = to_block_format_with_mask_bmm_one_mask(b, BK, BN)
                #print(a_mask_rowptrs, a_mask_cols)
                c = gen_empty_matrix_dense_blocks(M, N, BM, BN, batch_size=B)

                times = []
                ms = torch.inf
                too_slow_count = 0
                try:
                    for num_stages in stages:
                        for num_warps in warps:
                            if BM * BK * BN >= 128 * 128 * 32 and num_warps == 1:
                                continue
                            if BM * BK * BN >= 128 * 128 * 64 and num_warps <= 2:
                                continue
                            ms, _, _ = triton.testing.do_bench(lambda: mcsr_bmm_inner(a_mask_cols, a_block, b_block, c[1], num_warps, num_stages), rep=50)

                            if ms > torch_ms * 2:
                                too_slow_count += 1
                                if too_slow_count == 5:
                                    raise Exception('Too Slow') 
                            print(f'info: {num_stages} x {num_warps}, {ms:.4f}')
                            times.append((ms, BM, BK, BN, num_stages, num_warps))
                except Exception as e:
                    print('info: run triton failed ({BM} x {BK} x {BN})')
                    print(e)
                    #raise e
                verified = torch.allclose(c_ref, from_block_format(c[1]))
                print('info: verify passes:', verified)
                if verified:
                    times.sort(key=lambda x: x[0])
                    print(f'info: blocksparse mm: {times[0][0]:.4f} ({BM} x {BK} x {BN})')
                    sys.stdout.flush()
                    if times[0][0] < best_time:
                        best_time = times[0][0]

    print(f'{B}x{M}x{K}x{N}', f'{torch_ms:.4f}', f'{triton_ms:.4f}', f'{best_time:.4f}', sep='; ')
    sys.stdout.flush()
    
#test_random()

def test_post_shapes_lower_tri():
    shapes = [
        (32*16, 1024, 1024, 1024//16),
        (32*16, 1024, 1024, 4096//16),
        (32*16, 1024, 1024, 8192//16),
        (32*16, 2048, 2048, 1024//16),
        (32*16, 2048, 2048, 4096//16),
        (32*16, 2048, 2048, 8192//16),
    ]
    for shape in shapes:
        B, M, K, N = shape
        test_lower_triangular(B, M, K, N)


def test_single_batch():
    # shapes = [
    #     (1, 1024, 1024, 1024),
    #     (1, 1024, 1024, 4096),
    #     (1, 1024, 1024, 8192),
    #     (1, 2048, 2048, 1024//16),
    #     (1, 2048, 2048, 4096//16),
    #     (1, 2048, 2048, 8192//16),
    # ]
    
    shapes = [
        (1, 3072, 3072, 3072),
        (16, 3072, 3072, 3072),
        (64, 3072, 3072, 3072),
        (1, 4096, 4096, 4096),
    ]
    for shape in shapes:
        B, M, K, N = shape
        test_lower_triangular(B, M, K, N)




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', type=int, default=0)
parser.add_argument('-k', type=int)
parser.add_argument('-n', type=int)
parser.add_argument('-b', type=int)
args = parser.parse_args()

B, M, K, N = args.b, args.m, args.k, args.n

if M == 0:
    #test_single_batch()
    test_post_shapes_lower_tri()
else:
    test_lower_triangular(B, M, K, N)
