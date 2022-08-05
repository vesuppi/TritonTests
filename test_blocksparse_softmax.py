import sys
import torch
import triton 
from utils import *
from blocksparse_softmax_kernels import NewFormat, exp_1d, exp_2d

exp = exp_2d

def test_pointwise(B, M, N, log=sys.stdout):
    print(f'{B}x{M}x{N}')
    dtype = torch.float32
    a = torch.randn([B, M, N], dtype=dtype, device='cuda')
    BM = 16
    BN = BM
    a_data, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BN)

    a_mask = NewFormat(a_mask)
    b_mask, b_data = exp(a_mask, a_data)
    assert(torch.allclose(torch.exp(a_data), b_data))

    ms0 = triton.testing.do_bench(lambda: exp(a_mask, a_data))
    print(ms0)
    ms1 = triton.testing.do_bench(lambda: torch.exp(a))
    print(ms1)


N = 4096
test_pointwise(1, N, N)

# M = 4
# N = M
# dtype = torch.float32
# device = 'cuda'
# a = torch.rand((M, N), dtype=dtype, device=device)

# b = rowwise_softmax(a)
# print(a)
# print(b)

# b_ref = torch.softmax(a, axis=1)
# assert torch.allclose(b, b_ref), (b, b_ref)