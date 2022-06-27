import torch
import triton
import triton.language as tl 
import triton.testing
import torch.utils.benchmark as benchmark
from triton_matmul import matmul

import sys

# a = torch.arange(32*32, device='cuda')
# print(a)
# print(a.reshape([32, 32]))
# sys.exit(0)

M = 2048
K = M
N = M
BLOCK = 64

def cdiv(x, y):
    if x % y == 0:
        return x // y
    else:
        return x // y + 1

@triton.jit
def _kernel(a_ptr, b_ptr, c_ptr, M, N, K, t1, 
            BLOCK: tl.constexpr):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    a_start_addr = a_ptr + mid * BLOCK * K
    a_block_ptrs = a_start_addr + tl.arange(0, BLOCK * BLOCK)
    a_block_ptrs = tl.reshape(a_block_ptrs, (BLOCK, BLOCK))

    b_start_addr = b_ptr + nid * BLOCK * BLOCK
    b_block_ptrs = b_start_addr + tl.arange(0, BLOCK * BLOCK)
    b_block_ptrs = tl.reshape(b_block_ptrs, (BLOCK, BLOCK))

    c_start_addr = c_ptr + mid * BLOCK * N + nid * BLOCK * BLOCK
    c_block_ptrs = c_start_addr + tl.arange(0, BLOCK * BLOCK)
    c_block_ptrs = tl.reshape(c_block_ptrs, (BLOCK, BLOCK))

    c = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    for k in range(K//BLOCK):
        a = tl.load(a_block_ptrs)
        b = tl.load(b_block_ptrs)
        c += tl.dot(b, a)
        #c += a + b
        a_block_ptrs += BLOCK * BLOCK
        b_block_ptrs += BLOCK * N

        #tl.store(tl.reshape(t1 + tl.arange(0, BLOCK * BLOCK), (BLOCK, BLOCK)), a_block_ptrs)

    c = c.to(tl.float16)
    tl.store(c_block_ptrs, c)
    # a_rows = mid * BLOCK + tl.arange(0, BLOCK)
    # b_cols = nid * BLOCK + tl.arange(0, BLOCK)
    # c_ptrs = c_ptr + a_rows[:, None] * N + b_cols[None, :]
    # tl.store(c_ptrs, c)


def mm1(a, b):
    c = torch.empty([M, N], device=a.device, dtype=a.dtype)
    t1 = torch.empty([BLOCK, BLOCK], device=a.device, dtype=torch.int64)
    # grid = lambda META: (
    #     triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    # )
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    _kernel[grid](a, b, c, M, N, K, t1, BLOCK)
    return c


def to_block_format(a, BLOCK_M, BLOCK_N):
    M, N = a.shape
    b = torch.zeros(M*N, dtype=a.dtype, device=a.device)
    block_size = BLOCK_M * BLOCK_N
    i = 0
    for m in range(cdiv(M, BLOCK_M)):
        for n in range(cdiv(N, BLOCK_N)):
            block = a[m*BLOCK_M:(m+1)*BLOCK_M, n*BLOCK_N:(n+1)*BLOCK_N]
            b[i*block_size: (i+1)*block_size] = torch.flatten(block)
            i += 1            
    return b


def from_block_format(b, M, N, BLOCK_M, BLOCK_N):
    a = torch.zeros((M, N), dtype=b.dtype, device=b.device)
    block_size = BLOCK_M * BLOCK_N
    i = 0
    for m in range(cdiv(M, BLOCK_M)):
        for n in range(cdiv(N, BLOCK_N)):
            flat_block = b[i*block_size: (i+1)*block_size]
            a[m*BLOCK_M:(m+1)*BLOCK_M, n*BLOCK_N:(n+1)*BLOCK_N] = flat_block.reshape(BLOCK_M, BLOCK_N)
            i += 1
            
    return a



# a = torch.randn(4, 4, device='cuda', dtype=torch.float16)
# print(a)
# get_block_format(a, 2, 2)
# sys.exit(1)

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
#a = torch.arange(M*K, device='cuda', dtype=torch.float16).reshape(M, K)
b = torch.randn(K, N, device=a.device, dtype=a.dtype)
c = torch.mm(a, b)

a1 = to_block_format(a, BLOCK, BLOCK)
b1 = to_block_format(b, BLOCK, BLOCK)
c1 = mm1(a1, b1)
c2 = from_block_format(torch.flatten(c1), M, N, BLOCK, BLOCK)

torch_ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a, b))
triton_ms, _, _ = triton.testing.do_bench(lambda: mm1(a1, b1))
print(torch_ms, triton_ms, sep='; ')


def myprint(a):
    if len(a.shape) == 1:
        print(a)
        return
    M, N = a.shape
    for i in range(M):
        for j in range(N):
            print(f'{a[i,j]:.3f}', end=' ')
        print()
    print()

print('a1:')
print(a1)
print('b1:')
print(b1)


# torch.set_printoptions(edgeitems=8)
allclose = torch.allclose(c, c2, rtol=0.01, atol=0.5)
if allclose:
    print('allclose: True')
else:
    myprint(c)
    myprint(c2)

# print(c)
# print(c1)