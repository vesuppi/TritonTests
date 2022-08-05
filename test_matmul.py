import torch
import triton

from triton.ops.matmul import matmul


M = 512
K = M
N = M

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device=a.device, dtype=a.dtype)
c1 = torch.mm(a, b)
c2 = matmul(a, b)

print(c1)
print(c2)
print('allclose:', torch.allclose(c1, c2, rtol=0.01))
