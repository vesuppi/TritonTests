import sys
import torch

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


def to_block_format(a, BLOCK_M: int, BLOCK_N: int):
    M, N = a.shape
    outer_m_dim = cdiv(M, BLOCK_M)
    outer_n_dim = cdiv(N, BLOCK_N)
    inner_m_dim = BLOCK_M
    inner_n_dim = BLOCK_N

    res = torch.empty(
        (outer_m_dim, outer_n_dim, inner_m_dim, inner_n_dim),
        dtype=a.dtype,
        device=a.device,
    )

    # TODO - Implement/check for padding
    for outer_m in range(outer_m_dim):
        for outer_n in range(outer_n_dim):
            res[outer_m, outer_n, 0: inner_m_dim, 0: inner_n_dim] = a[
                outer_m * BLOCK_M: outer_m * BLOCK_M + inner_m_dim, outer_n * BLOCK_N: outer_n * BLOCK_N + inner_n_dim
            ]
    return res


def to_block_format_with_mask(a, BLOCK_M: int, BLOCK_N: int):
    M, N = a.shape
    outer_m_dim = cdiv(M, BLOCK_M)
    outer_n_dim = cdiv(N, BLOCK_N)
    inner_m_dim = BLOCK_M
    inner_n_dim = BLOCK_N

    res = torch.empty(
        (outer_m_dim, outer_n_dim, inner_m_dim, inner_n_dim),
        dtype=a.dtype,
        device=a.device,
    )

    mask = torch.ones([outer_m_dim, outer_n_dim], device=a.device)

    # TODO - Implement/check for padding
    for m in range(outer_m_dim):
        for n in range(outer_n_dim):
            block = a[
                m * BLOCK_M: (m+1) * BLOCK_M, 
                n * BLOCK_N: (n+1) * BLOCK_N
            ]
            res[m, n, 0: BLOCK_M, 0: BLOCK_N] = block
            if torch.count_nonzero(block) == 0:
                mask[m, n] = 0
    return (res, mask)


def from_block_format(a):
    # TODO - Implement/check for padding
    outer_m_dim, outer_n_dim, BLOCK_M, BLOCK_N = a.shape

    M = outer_m_dim * BLOCK_M
    N = outer_n_dim * BLOCK_N

    res = torch.zeros((M, N), dtype=a.dtype, device=a.device)

    for outer_m in range(outer_m_dim):
        for outer_n in range(outer_n_dim):
            res[outer_m * BLOCK_M: outer_m * BLOCK_M + BLOCK_M, outer_n * BLOCK_N: outer_n * BLOCK_N + BLOCK_N] = a[
                    outer_m, outer_n, 0: BLOCK_M, 0: BLOCK_N
                ]
    return res


def get_lower_triangular_mask(m, n):
    mask = torch.tril(torch.ones([m, n], device='cuda'))
    #mask = torch.ones([m, n], device='cuda')
    return mask


def to_csr_ptrs(a, device='cuda'):
    m, n = a.shape
    nnz = 0
    for i in range(m):
        for j in range(n):
            if a[i,j] != 0:
                nnz += 1
    
    rowptrs = torch.zeros(m+1, dtype=torch.int, device=device)
    rowptrs[0] = 0
    cols = torch.zeros(nnz, dtype=torch.int, device=device)
    nnz = 0
    for i in range(m):
        for j in range(n):
            if a[i,j] != 0:
                cols[nnz] = j
                nnz += 1
        rowptrs[i+1] = nnz
    assert nnz == torch.sum(a)
    return (rowptrs, cols)


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


def gen_lower_triangular_mcsr_matrix(M, N, BM, BN, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.ones([m, n], dtype=torch.int, device=device)
    data = torch.randn([m, n, BM, BN], dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            if j > i:
                data[i,j] = torch.zeros(BM, BN)
                mask[i,j] = 0

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

