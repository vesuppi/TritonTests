import sys
from numpy import format_float_positional
import torch
import triton
import triton.language as tl
import triton.testing

torch.backends.cuda.matmul.allow_tf32 = True

#@torch.jit.script
def ceil_div(x: int, y: int):
    return (x + y - 1) // y

#@torch.jit.script
def to_block_format(a, BLOCK_M: int, BLOCK_N: int):
    M, N = a.shape
    outer_m_dim = ceil_div(M, BLOCK_M)
    outer_n_dim = ceil_div(N, BLOCK_N)
    inner_m_dim = BLOCK_M
    inner_n_dim = BLOCK_N

    res = torch.zeros(
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
            # for inner_m in range(inner_m_dim):
            #     # res[outer_m, outer_n, inner_m, 0: inner_n_dim] = a[
            #     #         outer_m * BLOCK_M + inner_m, outer_n * BLOCK_N: outer_n * BLOCK_N + inner_n_dim
            #     #     ]
            #     for inner_n in range(inner_n_dim):
            #         res[outer_m, outer_n, inner_m, inner_n] = a[
            #             outer_m * BLOCK_M + inner_m, outer_n * BLOCK_N + inner_n
            #         ]
    return res


def from_block_format(a, BLOCK_M, BLOCK_N):
    # TODO - Implement/check for padding
    outer_m_dim, outer_n_dim, inner_m_dim, inner_n_dim = a.shape

    M = outer_m_dim * BLOCK_M
    N = outer_n_dim * BLOCK_N

    res = torch.zeros((M, N), dtype=a.dtype, device=a.device)

    for outer_m in range(outer_m_dim):
        for outer_n in range(outer_n_dim):
            res[outer_m * BLOCK_M: outer_m * BLOCK_M + inner_m_dim, outer_n * BLOCK_N: outer_n * BLOCK_N + inner_n_dim] = a[
                    outer_m, outer_n, 0: inner_m_dim, 0: inner_n_dim
                ]
            # for inner_m in range(inner_m_dim):
            #     for inner_n in range(inner_n_dim):
            #         res[outer_m * BLOCK_M + inner_m, outer_n * BLOCK_N + inner_n] = a[
            #             outer_m, outer_n, inner_m, inner_n
            #         ]
    return res


@triton.jit
def _kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    num_blocks_in_K: tl.constexpr,
    num_blocks_in_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute the outer_m, outer_n
    outer_m = tl.program_id(0)
    outer_n = tl.program_id(1)

    a_block_offset = outer_m * num_blocks_in_K * BLOCK_M * BLOCK_K
    a_block_offset = tl.multiple_of(a_block_offset, BLOCK_M * BLOCK_K)


    x1_ptr = tl.arange(0, BLOCK_M)
    y1_ptr = tl.arange(0, BLOCK_K)
    a_block_offsets = a_block_offset + x1_ptr[:, None] * BLOCK_K + y1_ptr[None, :]
    #a_block_offsets1 = tl.max_contiguous(tl.multiple_of(a_block_offsets, BLOCK_M*BLOCK_K), BLOCK_M*BLOCK_K)
    a_block_ptrs = a_ptr + a_block_offsets

    #a_block_ptrs = tl.max_contiguous(a_block_ptrs, BLOCK_M*BLOCK_K)

    b_start_addr = outer_n * BLOCK_K * BLOCK_N
    b_start_addr = tl.multiple_of(b_start_addr, BLOCK_N * BLOCK_K)
    x2_ptr = tl.arange(0, BLOCK_K)
    y2_ptr = tl.arange(0, BLOCK_N)
    b_block_offsets = b_start_addr + x2_ptr[:, None] * BLOCK_N + y2_ptr[None, :]
    #b_block_offsets1 = tl.max_contiguous(tl.multiple_of(b_block_offsets, BLOCK_N*BLOCK_K), BLOCK_N*BLOCK_K)
    b_block_ptrs = b_ptr + b_block_offsets

    #b_block_ptrs = tl.max_contiguous(b_block_ptrs, BLOCK_K*BLOCK_N)

    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(num_blocks_in_K):
        a = tl.load(a_block_ptrs)
        b = tl.load(b_block_ptrs)
        c += tl.dot(a, b)
        a_block_ptrs += BLOCK_M * BLOCK_K
        b_block_ptrs += BLOCK_K * (BLOCK_N * num_blocks_in_N)

    # c = c.to(tl.float16)

    c_start_addr = c_ptr + (outer_m * num_blocks_in_N + outer_n) * BLOCK_M * BLOCK_N
    x3_ptr = tl.arange(0, BLOCK_M)
    y3_ptr = tl.arange(0, BLOCK_N)
    c_block_ptrs = c_start_addr + x3_ptr[:, None] * BLOCK_N + y3_ptr[None, :]
    #c_block_ptrs = tl.max_contiguous(c_block_ptrs, BLOCK_M*BLOCK_N)
    tl.store(c_block_ptrs, c)


def blocked_mm(a, b):
    outer_m_dim, outer_k_dim, BLOCK_M, BLOCK_K = a.shape
    outer_k_dim, outer_n_dim, BLOCK_K, BLOCK_N = b.shape

    M = outer_m_dim * BLOCK_M
    N = outer_n_dim * BLOCK_N
    K = outer_k_dim * BLOCK_K

    c = torch.zeros(
        (outer_m_dim, outer_n_dim, BLOCK_M, BLOCK_N), device=a.device, dtype=a.dtype
    )
    grid = (outer_m_dim, outer_n_dim)
    _kernel[grid](a, b, c, M, N, K, outer_k_dim, outer_n_dim, BLOCK_M, BLOCK_N, BLOCK_K)
    return c


def check_block_format_utils():
    BLOCK = 16
    a = torch.randn(64, 64)
    blocked_a = to_block_format(a, BLOCK, BLOCK)
    dense_a = from_block_format(blocked_a, BLOCK, BLOCK)
    assert torch.allclose(a, dense_a)


def run_triton(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N):
    ref_c = torch.mm(a, b)
    # Triton
    blocked_a = to_block_format(a, BLOCK_M, BLOCK_K)
    blocked_b = to_block_format(b, BLOCK_K, BLOCK_N)
    blocked_c = blocked_mm(blocked_a, blocked_b)
    res_c = from_block_format(blocked_c, BLOCK_M, BLOCK_N)

    tol = 1e-1
    assert torch.allclose(ref_c, res_c, rtol=0.05, atol=tol)

    # Benchmarking
    for _ in range(3):                                    
        blocked_mm(blocked_a, blocked_b)

    ms, _, _ = triton.testing.do_bench(lambda: blocked_mm(blocked_a, blocked_b))
    return ms

def run_torch(a, b, M, K, N):
    for _ in range(5):
        torch.mm(a, b)
    ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a, b))
    return ms


def check_triton_mm():
    torch.manual_seed(0)
    M = 64
    K = 128
    N = 256
    BLOCK = 32

    dtype = torch.float16
    for M in [2048, 4096]:
        for N in [2048, 4096]:
            for K in [2048, 4096]:
            #for K in [4096*2]:
                a = torch.randn((M, K), device="cuda", dtype=dtype)
                b = torch.randn((K, N), device="cuda", dtype=dtype)

                ms1 = run_torch(a, b, M, K, N)
                ms2 = run_triton(a, b, M, K, N, 64, 64, 64)
                FLOPS1 = 2 * M * K * N / ms1 / 10**9
                FLOPS2 = 2 * M * K * N / ms2 / 10**9
                print(M, K, N, ms1, FLOPS1, FLOPS1/312, ms2, FLOPS2, FLOPS2/312)
    sys.exit(0)

    # for shape in [(128, 4096, 1000)]:
    #     M, K, N = shape
    #     a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    #     b = torch.randn((K, N), device="cuda", dtype=a.dtype)
    #     ms1 = run_torch(a, b, M, K, N)
    #     for blocks in [(64, 64, 64)]:
    #         BLOCK_M, BLOCK_K, BLOCK_N = blocks
    #         ms2 = run_triton(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
    #         print(ms1, ms2)

    # return
    i = 0
    for dtype in [torch.float16, torch.float32]:
        for M in [512, 1024, 64, 128, 256]:
            for N in [512, 1024, 64, 128, 256]:
                for K in [512, 1024, 64, 128, 256]:
                    a = torch.randn((M, K), device="cuda", dtype=dtype)
                    b = torch.randn((K, N), device="cuda", dtype=dtype)

                    ms1 = run_torch(a, b, M, K, N)

                    triton_times = []

                    for BLOCK_M in [64, 128, 32]:
                        for BLOCK_K in [32, 64, 128]:
                            for BLOCK_N in [64, 128, 32]:
                                if BLOCK_M > M or BLOCK_K > K or BLOCK_N > N:
                                    continue
                                
                                ms2 = run_triton(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
                                triton_times.append((ms2, (BLOCK_M, BLOCK_K, BLOCK_N)))

                    triton_times.sort(key=lambda x: x[0])
                    
                    print(f'{M} x {K} x {N}', end='; ')
                    print(f'{ms1:.4f}', end='; ')
                    for i in range(5):
                        ms, blocks = triton_times[i]
                        print(f'{ms:.4f}; {blocks}', end='; ')
                    print()


if __name__ == "__main__":
    check_block_format_utils()
    check_triton_mm()




