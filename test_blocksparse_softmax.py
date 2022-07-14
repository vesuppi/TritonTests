import sys
import torch
print('imported torch')
import triton 
from utils import *
from blocksparse_softmax_kernels import rowwise_softmax, softmax

M = 4
N = M
dtype = torch.float32
device = 'cuda'
a = torch.rand((M, N), dtype=dtype, device=device)

b = rowwise_softmax(a)
print(a)
print(b)

b_ref = torch.softmax(a, axis=1)
assert torch.allclose(b, b_ref), (b, b_ref)