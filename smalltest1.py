import sys
import torch
print('imported torch')
import triton 
import triton.language as tl
import utils
from utils import *

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