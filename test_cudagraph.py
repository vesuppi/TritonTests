
from triton.ops.matmul import matmul
import torch
from torch.utils.benchmark import Timer


def time_with_torch_timer(fn, args, kwargs={}, iters=100):
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    # Measure end-to-end time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.timeit(iters)

    return tt.mean * 1000


def torch_foo(a, b):
    return torch.mm(a, b)


def triton_foo(a, b):
    return matmul(a, b)


N = 2048
a = torch.randn((N, N), device='cuda', dtype=torch.float16)
b = torch.randn((N, N), device='cuda', dtype=a.dtype)
c = torch.randn((N, N), device='cuda', dtype=a.dtype)

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        d1 = triton_foo(a, b)
        d2 = torch_foo(a, b)
        print(d1[0,0])
        print(d2[0,0])
        assert(torch.allclose(d1, d2, atol=0.1, rtol=0.01))

torch.cuda.current_stream().wait_stream(s)


g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    d1 = triton_foo(a, b)

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    d2 = torch_foo(a, b)

real_a = torch.rand_like(a)
real_b = torch.rand_like(b)

a.copy_(real_a)
b.copy_(real_b)

ms = time_with_torch_timer(g1.replay, ())
print(ms)

ms = time_with_torch_timer(g2.replay, ())
print(ms)