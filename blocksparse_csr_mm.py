import sys
import torch
import triton 
import triton.language as tl

class BCSR():
    def __init__(self, rowptrs, cols, vals) -> None:
        self.rowptrs = rowptrs
        self.cols = cols
        self.vals = vals


class MCSR():
    def __init__(self, rowptrs, cols, vals) -> None:
        self.rowptrs = rowptrs
        self.cols = cols
        self.vals = vals


def cdiv(x, y):
    return (x + y -1) // y


def gen_random_matrix(M, N, BM, BN, density=0.5, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.zeros([m, n], dtype=torch.int, device=device)
    for i in range(m):
        for j in range(n):
            p = torch.rand(1)
            if p[0] < density:
                mask[i,j] = 1
    #print(mask)
    nnz = torch.sum(mask)
    data = torch.randn([nnz, BM, BN], dtype=dtype, device=device)
    return (mask, data)


def gen_random_matrix_dense_blocks(M, N, BM, BN, density=0.5, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.ones([m, n], dtype=torch.int, device=device)
    data = torch.randn([m, n, BM, BN], dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            p = torch.rand(1)
            if p[0] > density:
                mask[i,j] = 0
                data[i,j] = torch.zeros(BM, BN)

    return (mask, data)


def gen_random_mcsr_matrix(M, N, BM, BN, density=1, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.zeros([m, n], dtype=torch.int, device=device)
    data = torch.randn([m, n, BM, BN], dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            p = torch.rand(1)
            if p[0] < density:
                mask[i,j] = 1
            else:
                data[i,j] = torch.zeros(BM, BN)

    nnz = torch.sum(mask)
    rowptrs = torch.zeros(m+1, dtype=torch.int, device=device)
    rowptrs[0] = 0
    cols = torch.zeros(nnz, dtype=torch.int, device=device)
    
    nnz = 0
    for i in range(m):
        for j in range(n):
            if mask[i,j] != 0:
                cols[nnz] = j
                nnz += 1
        rowptrs[i+1] = nnz
    assert nnz == torch.sum(mask)
    return BCSR(rowptrs, cols, data)


def gen_empty_matrix_dense_blocks(M, N, BM, BN, density=0.5, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.ones([m, n], dtype=torch.int, device=device)
    data = torch.empty([m, n, BM, BN], dtype=dtype, device=device)
    return (mask, data)


def from_block_format(a):
    # TODO - Implement/check for padding
    outer_m_dim, outer_n_dim, BM, BN = a.shape

    M = outer_m_dim * BM
    N = outer_n_dim * BN

    res = torch.zeros((M, N), dtype=a.dtype, device=a.device)

    for outer_m in range(outer_m_dim):
        for outer_n in range(outer_n_dim):
            res[outer_m * BM: outer_m * BM + BM, \
                outer_n * BN: outer_n * BN + BN] = \
            a[outer_m, outer_n, 0: BM, 0: BN]
    return res

@triton.jit
def _kernel_sss(a, b, num_ks, ks, buf):
    mid = tl.program_id(0)
    K = tl.load(num_ks+mid)
    next_K = tl.load(num_ks+mid+1)
    i = 0
    for kp in range(K, next_K):
        k = tl.load(ks+kp)
        tl.store(buf+mid*4+i, k)
        i += 1
        # C[mid, nid] = A[kp] * B[k, nid]


def matmul_sss(a, b):
    M = 16
    K = M
    N = K
    BM = 4
    BK = BM
    BN = BM
    grid = (cdiv(M, BM), cdiv(N, BN))
    num_ks = torch.tensor([0,1,3,6,10], device='cuda')
    ks = torch.tensor([
            0,
            0, 1,
            0, 1, 2,
            0, 1, 2, 3,
        ], device='cuda')
    buf = torch.zeros(64, device='cuda')
    _kernel_sss[grid](a, b, num_ks, ks, buf)
    print(buf)



def test1():
    a = gen_random_matrix(8, 8, 2, 2)
    b = gen_random_matrix(8, 8, 2, 2)
    matmul_sss(a[1], b[1])


@triton.jit
def _kernel_mcsr_mm(a_rowptrs, a_cols, a_vals, b_vals, c_vals, 
                                BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
                                nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
                                buf
                                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    k_start = tl.load(a_rowptrs)
    k_end = tl.load(a_rowptrs+1)
    c = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(nBK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)

        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN

    # for kp in range(k_start, k_end):
    #     k = tl.load(a_cols+kp)
    #     a = tl.load(a_ptrs+a_block_size*k)
    #     b = tl.load(b_ptrs+b_block_size * nBN*k)
    #     c += tl.dot(a, b)

    #     # a_ptrs += a_block_size
    #     # b_ptrs += b_block_size * nBN


    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(c_ptrs, c)


def mcsr_mm(a: MCSR, b: MCSR, num_warps=4, num_stages=3):
    nBM, nBK, BM, BK = a.vals.shape
    nBK, nBN, BK, BN = b.vals.shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN
    c = gen_empty_matrix_dense_blocks(M, N, BM, BN)

    grid = (nBM, nBN)
    #print(grid)
    buf = torch.zeros(64, device='cuda')
    _kernel_mcsr_mm[grid](a.rowptrs, a.cols, a.vals, b.vals, c[1],
                                    BM, BK, BN, nBM, nBK, nBN, buf,
                                    num_warps=num_warps, num_stages=num_stages
                                    )
    return c


def verify_run():
    M = 32
    K = M
    N = M

    BM = 16
    BK = BM
    BN = BM
    a = gen_random_mcsr_matrix(M, K, BM, BK, density=1)
    b = gen_random_mcsr_matrix(K, N, BK, BN, density=1)
    a_ref = from_block_format(a.vals)
    b_ref = from_block_format(b.vals)
    c_ref = torch.mm(a_ref, b_ref)
    c = mcsr_mm(a, b)
    print('verify passes:', torch.allclose(c_ref, from_block_format(c[1])))


def benchmark_run():
    M = 2048
    K = M
    N = M
    
    BMs = [16, 32, 64]
    BKs = [16, 32, 64]
    BNs = [16, 32, 64]
    stages = [1,2,3,4,5]
    warps = [1,2,4,8]

    TEST_RUN = True

    if TEST_RUN:
        BMs, BKs, BNs = [32], [16], [32]
        stages, warps = [1,2,3,4,5], [1,2,4]

    times = []
    for BM in BMs:
        for BK in BKs:
            for BN in BNs:
                a = gen_random_mcsr_matrix(M, K, BM, BK, density=1)
                b = gen_random_mcsr_matrix(K, N, BK, BN, density=1)
                a_ref = from_block_format(a.vals)
                b_ref = from_block_format(b.vals)
                c_ref = torch.mm(a_ref, b_ref)
                ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a_ref, b_ref))
                print(f'torch mm: {ms:.4f}')
                
                for num_stages in stages:
                    for num_warps in warps:
                        ms, _, _ = triton.testing.do_bench(lambda: mcsr_mm(a, b, num_warps, num_stages))
                        times.append((ms, BM, BK, BN, num_stages, num_warps))
                times.sort(key=lambda x: x[0])
                print(f'blocksparse mm: {times[0][0]:.4f}')

    

verify_run()
benchmark_run()