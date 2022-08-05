import torch
import pytest
import triton
import sys



def bench_matmul(M, N, K, block, layout_mode, op_mode, AT, BT, dtype, warmup=100, rep=1000):
    Z, H = 1, 1
    make_layout = {
        'tril': lambda H, M, N: torch.tril(torch.ones((H, M, N), dtype=torch.int64)),
        'dense': lambda H, M, N: torch.ones(H, M, N, dtype=torch.int64),
    }[layout_mode]
    # create layout
    shape = {'sdd': (M, N), 'dsd': (K, M) if AT else (M, K), 'dds': (N, K) if BT else (K, N)}[op_mode]
    layout = make_layout(H, shape[0] // block, shape[1] // block)
    # creat inputs
    a = torch.randn((Z, H, K, M) if AT else (Z, H, M, K), dtype=dtype, device='cuda')
    b = torch.randn((Z, H, N, K) if BT else (Z, H, K, N), dtype=dtype, device='cuda')
    # create op
    tflops = lambda ms: num_flops / ms * 1e3

    print(a.shape)
    print(layout)

    op = triton.ops.blocksparse.matmul(layout, block, op_mode, device="cuda", trans_a=AT, trans_b=BT)
    # inputs
    a = triton.testing.sparsify_tensor(a, layout, block) if op_mode == 'dsd' else a
    b = triton.testing.sparsify_tensor(b, layout, block) if op_mode == 'dds' else b


    mean_ms, min_ms, max_ms = triton.testing.do_bench(lambda: op(a, b), warmup=warmup, rep=rep)
    num_flops = {
        'sdd': 2 * Z * K * float(layout.sum()) * block * block,
        'dsd': 2 * Z * N * float(layout.sum()) * block * block,
        'dds': 2 * Z * M * float(layout.sum()) * block * block
    }[op_mode] * 1e-12
    return tflops(mean_ms), tflops(min_ms), tflops(max_ms)

M = 64
N = M
K = M

bench_matmul(M, N, K, 16, 'tril', 'dsd', False, False, torch.float16)


# def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=3, H=2, M=512, N=384, K=256):
#     seed = 0
#     torch.manual_seed(seed)
#     is_sdd = MODE == "sdd"
#     is_dsd = MODE == "dsd"
#     is_dds = MODE == "dds"
#     do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
#     do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
#     # create inputs
#     # create op
#     a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
#     b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
#     c_shape = (Z, H, M, N)
#     shape = {
#         "sdd": (M, N),
#         "dsd": (a_shape[2], a_shape[3]),
#         "dds": (b_shape[2], b_shape[3]),
#     }[MODE]
#     layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
#     layout[1, 2, :] = 0
#     layout[1, :, 1] = 0
#     # create data
#     a_ref, a_tri = triton.testing.make_pair(a_shape, alpha=.1)
#     b_ref, b_tri = triton.testing.make_pair(b_shape, alpha=.1)
#     dc_ref, dc_tri = triton.testing.make_pair(c_shape)
#     print(a_ref, a_tri)
#     # compute [torch]
#     dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
#     a_ref = do_mask(a_ref) if is_dsd else a_ref
#     b_ref = do_mask(b_ref) if is_dds else b_ref
#     a_ref.retain_grad()
#     b_ref.retain_grad()
#     c_ref = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref,
#                          b_ref.transpose(2, 3) if TRANS_B else b_ref)
#     c_ref.backward(dc_ref)
#     c_ref = do_sparsify(c_ref) if is_sdd else c_ref
#     da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
#     db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
#     # triton result
#     dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
#     a_tri = do_sparsify(a_tri) if is_dsd else a_tri
#     b_tri = do_sparsify(b_tri) if is_dds else b_tri
#     a_tri.retain_grad()
#     b_tri.retain_grad()
#     op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda")
#     c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)
#     triton.testing.catch_oor(lambda: c_tri.backward(dc_tri), pytest)
#     da_tri = a_tri.grad
#     db_tri = b_tri.grad
#     # compare
#     triton.testing.assert_almost_equal(c_ref, c_tri)
#     triton.testing.assert_almost_equal(da_ref, da_tri)
#     triton.testing.assert_almost_equal(db_ref, db_tri)


# if __name__ == '__main__':
#     for mode in ["sdd", "dds", "dsd"]:
#         for trans_A in [False, True]:
#             for trans_B in [False, True]:
#                 for block in [16, 32, 64]:
#                     for dtype in [torch.float16]:
#                         test_matmul(mode, trans_A, trans_B, block, dtype)
#                         sys.exit(1)