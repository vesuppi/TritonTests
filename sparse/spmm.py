import numpy as np
import torch
import scipy
import scipy.sparse
import torchdynamo
from microbenchmarks.benchmark_helper import time_with_torch_timer
import triton

from triton.ops.blocksparse import matmul as blocksparse_matmul
from triton.ops.matmul import matmul as triton_matmul

def torch_spmm(a: torch.sparse_coo_tensor, b):
    return torch.sparse.mm(a, b)

@torchdynamo.optimize('inductor')
def inductor_spmm_triton(a: torch.sparse_coo_tensor, b):
    return torch.sparse.mm(a, b)

'''Code gen example by inductor
a_mask, a_data = convert_to_triton_format(a)
triton_spmm = triton.ops.blocksparse.matmul(a_mask, ...)
c = ...
triton_spmm(a_data, b, out=c)
'''    


def get_random_coo_matrix(M, K, density, dtype=torch.float16, device='cuda'):
    a = scipy.sparse.random(M, K, density=density)
    a_tensor = torch.sparse_coo_tensor(
            np.vstack((a.row, a.col)),
            a.data,
            a.shape,
            dtype=dtype,
            device=device
        )
    return a_tensor


def get_mask_and_data(a: torch.sparse_coo_tensor, blocksize):
    assert len(blocksize) == 2
    a_cpu = a.cpu()
    a_coo = scipy.sparse.coo_matrix(
        (
            a_cpu._values().numpy(), 
            (a_cpu._indices()[0].numpy(), a_cpu._indices()[1].numpy())
        )
    )

    a_bsr = a_coo.tobsr(blocksize=blocksize)
    # print(a_bsr.indptr)
    # print(a_bsr.indices)

    M, N = a_bsr.shape
    BLOCK_SIZE_M, BLOCK_SIZE_N = blocksize
    mask = torch.zeros((M//BLOCK_SIZE_M, N//BLOCK_SIZE_N), dtype=torch.int)
    
    indptr = a_bsr.indptr
    cols = a_bsr.indices
    for r in range(len(indptr)-1):
        for p in range(indptr[r], indptr[r+1]):
            c = cols[p]
            mask[r, c] = 1
    
    data = torch.from_numpy(a_bsr.data).cuda()
    return (mask.cuda(), data)



M = 1024
K = 512
N = M

device = 'cuda'
dtype = torch.float32  # check if it works for triton and use float32 for torch
a = get_random_coo_matrix(M, K, 0.25, dtype=dtype, device=device)
b = torch.randn(K, N, dtype=dtype, device=device)
c = torch_spmm(a, b)


block = 32
mask, data = get_mask_and_data(a, (block, block))


#print(data.shape)

#data = triton.testing.sparsify_tensor(a.to_dense(), mask, blocksize[0])
#print(mask)

mask = mask[None, :]
data = data[None, :]
b1 = b[None, None, :, :]

#print(data.shape)
#print(c.shape)

triton_spmm = blocksparse_matmul(
                layout=mask,
                block=block,
                mode="dsd",
                device="cuda",
                trans_a=False,
                trans_b=False,
            )

c1 = torch.squeeze(triton_spmm(data, b1))
if not torch.allclose(c, c1, rtol=0.01, atol=0.01):
    print(c[0])
    print(c1[0])

a_dense = a.to_dense()

torch_sparse_ms, _, _ = triton.testing.do_bench(lambda: torch_spmm(a, b))
torch_dense_ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a_dense, b))
triton_ms, _, _ = triton.testing.do_bench(lambda: triton_spmm(data, b1))
triton_dense_ms, _, _ = triton.testing.do_bench(lambda: triton_matmul(a_dense, b))
print(torch_sparse_ms, torch_dense_ms, triton_ms, triton_dense_ms, sep='; ')

# torch_sparse_ms = time_with_torch_timer(torch_spmm, (a, b)).mean * 1000
# torch_dense_ms = time_with_torch_timer(torch.mm, (a_dense, b)).mean * 1000
# triton_ms = time_with_torch_timer(triton_spmm, (data, b1)).mean * 1000
# triton_dense_ms = time_with_torch_timer(triton_matmul, (a_dense, b)).mean * 1000
# print(torch_sparse_ms, torch_dense_ms, triton_ms, triton_dense_ms, sep='; ')

'''
# Takeaways
- This spmm implementation is highly specialied (not general at all)

- torch.spmm is really slow. >10x slower than dense torch.mm for around 80% sparsity

- If use random sparse matrix, it's unlikely to get an empty block
unless the sparsity if really low (not a single nonzero in the block)

So this format is better suited for structured sparsity.

Is GNN unstructured sparsity?

Three major sources of sparsity
- Pruned weights
- Attention (structured)
- GNN

- Triton's SpMM seems much faster than its matmul when totally dense?


'''