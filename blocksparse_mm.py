
import torch
import triton 
import triton.language as tl

print(triton.__file__)

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


def gen_empty_matrix_dense_blocks(M, N, BM, BN, density=0.5, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.empty([m, n], dtype=torch.int, device=device)
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
def _kernel_mm_mask_dense_block(a_mask, a_data, b_mask, b_data, c_mask, c_data, 
                                BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
                                nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
                                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_data + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_data + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    i = 0
    c = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(nBK):
        # a_has_block = tl.load(a_mask + m*nBK + k)
        # b_has_block = tl.load(b_mask + k*nBN + n)
        # if a_has_block & b_has_block:
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)

        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN


    c = c.to(tl.float16)

    c_ptrs = c_data + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(c_ptrs, c)


def mm_mask_dense_block(a, b, num_warps=4, num_stages=3):
    nBM, nBK, BM, BK = a[1].shape
    nBK, nBN, BK, BN = b[1].shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN
    c = gen_empty_matrix_dense_blocks(M, N, BM, BN)

    grid = (nBM, nBN)
    #print(grid)
    #buf = torch.zeros(64, device='cuda')
    _kernel_mm_mask_dense_block[grid](a[0], a[1], b[0], b[1], c[0], c[1],
                                    BM, BK, BN, nBM, nBK, nBN, 
                                    num_warps=num_warps, num_stages=num_stages
                                    )
    return c


def test2():
    M = 1024
    N = M
    K = M
    BM = 64
    BK = 32
    BN = BM
    a = gen_random_matrix_dense_blocks(M, K, BM, BK, density=1)
    b = gen_random_matrix_dense_blocks(K, N, BK, BN, density=1)
    a_ref = from_block_format(a[1])
    b_ref = from_block_format(b[1])
    c_ref = torch.mm(a_ref, b_ref)
    print(c_ref)
  
    c = mm_mask_dense_block(a, b)
    print('verify passes:', torch.allclose(c_ref, from_block_format(c[1])))

    BMs = [16, 32, 64]
    BKs = [16, 32, 64]
    BNs = [16, 32, 64]
    stages = [1,2,3,4,5]
    warps = [1,2,4,8]

    TEST_RUN = True

    if TEST_RUN:
        BMs, BKs, BNs = [32], [32], [32]
        stages, warps = [1,2,3,4,5], [1,2,4]

    times = []
    for BM in BMs:
        for BK in BKs:
            for BN in BNs:


                for num_stages in stages:
                    for num_warps in warps:
                        ms, _, _ = triton.testing.do_bench(lambda: mm_mask_dense_block(a, b, num_warps, num_stages))
                        times.append((ms, BM, BK, BN, num_stages, num_warps))
    times.sort(key=lambda x: x[0])
    print(f'blocksparse mm: {times[0][0]:.4f}')

    ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a_ref, b_ref))
    print(f'torch mm: {ms:.4f}')



def verify_run():
    M = 32
    K = M
    N = M

    BM = 16
    BK = BM
    BN = BM
    a = gen_random_matrix_dense_blocks(M, K, BM, BK, density=1)
    b = gen_random_matrix_dense_blocks(K, N, BK, BN, density=1)
    a_ref = from_block_format(a[1])
    b_ref = from_block_format(b[1])
    c_ref = torch.mm(a_ref, b_ref)
    c = mm_mask_dense_block(a, b)
    print('verify passes:', torch.allclose(c_ref, from_block_format(c[1])))


def benchmark_run():
    M = 1024
    K = 2048
    N = M
    
    BMs = [64, 128]
    BKs = [32, 64, 128]
    BNs = [32, 64, 128]
    stages = [1,2,3,4,5]
    warps = [1,2,4,8]

    TEST_RUN = False

    if TEST_RUN:
        BMs, BKs, BNs = [32], [16], [32]
        stages, warps = [1,2,3,4,5], [1,2,4]

    times = []
    for BM in BMs:
        for BK in BKs:
            for BN in BNs:
                if (BM == 128 and BK == 128) or (BM == 128 and BN == 128) or (BN == 128 and BK == 128):
                    continue

                a = gen_random_matrix_dense_blocks(M, K, BM, BK, density=1)
                b = gen_random_matrix_dense_blocks(K, N, BK, BN, density=1)
                a_ref = from_block_format(a[1])
                b_ref = from_block_format(b[1])
                c_ref = torch.mm(a_ref, b_ref)
                ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a_ref, b_ref))
                print(f'torch mm: {ms:.4f}')
                c = mm_mask_dense_block(a, b)
                print('verify passes:', torch.allclose(c_ref, from_block_format(c[1])))

                for num_stages in stages:
                    for num_warps in warps:
                        ms, _, _ = triton.testing.do_bench(lambda: mm_mask_dense_block(a, b, num_warps, num_stages))
                        times.append((ms, BM, BK, BN, num_stages, num_warps))
                times.sort(key=lambda x: x[0])
                print(f'blocksparse mm: {times[0][0]:.4f} ({BM} x {BK} x {BN})')

    

#verify_run()
benchmark_run()