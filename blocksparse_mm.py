
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
def _kernel_mm_mask_dense_block(a_mask, a_data, b_mask, b_data, c_mask, c_data, 
                                BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
                                nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
                                buf
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
        a_has_block = tl.load(a_mask + m*nBK + k)
        b_has_block = tl.load(b_mask + k*nBN + n)
        if a_has_block & b_has_block:
            # Load the (m,k) tile from A and (k,n) tile from B
            a = tl.load(a_ptrs + a_block_size * k)
            b = tl.load(b_ptrs + b_block_size * k * nBN)
            c += tl.dot(a, b)

            # if (m == 0) & (n == 0):
            #     tl.store(buf+i, k)
            #     i += 1

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
    buf = torch.zeros(64, device='cuda')
    _kernel_mm_mask_dense_block[grid](a[0], a[1], b[0], b[1], c[0], c[1],
                                    BM, BK, BN, nBM, nBK, nBN, buf,
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
    print(torch.allclose(c_ref, from_block_format(c[1])))

    times = []
    for num_stages in [1,2,3,4,5]:
        for num_warps in [1,2,4,8]:
            ms, _, _ = triton.testing.do_bench(lambda: mm_mask_dense_block(a, b, num_warps, num_stages))
            times.append((ms, num_stages, num_warps))
    times.sort(key=lambda x: x[0])
    print(f'blocksparse mm: {times[0][0]:.4f}')

    ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a_ref, b_ref))
    print(f'torch mm: {ms:.4f}')

test2()