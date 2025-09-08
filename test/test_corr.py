import torch
import torch.nn.functional as F
from local_corr import local_corr

def baseline_corr(im_A: torch.Tensor, im_B: torch.Tensor):
    out = torch.einsum("bnc, bhwc -> bnhw", im_A, im_B)
    return out

device = "cuda"
B, HW, H, W, C, N = 8, 64*64, 64, 64, 1024, 64
im_A = torch.randn(B, HW, C).to(device)
im_B = torch.randn(B, H, W, C).to(device)
cols = torch.randint(0, W, (B, HW, N)).to(device)
rows = torch.randint(0, H, (B, HW, N)).to(device)


T = 100
import time

def perf(f, name):
    with torch.inference_mode():
        f(im_A,im_B)
        t0 = time.perf_counter()
        for t in range(T):
            f(im_A,im_B)
        print(f"Time: {(time.perf_counter()-t0)/T} for {name}")

naive_compiled = torch.compile(baseline_corr)
# perf(baseline_corr, "baseline")
# perf(corr, "custom")

perf(baseline_corr,"naive_corr")
# perf(naive_compiled,"naive_corr_compiled")
perf(local_corr,"custom_corr")


# print(torch.einsum("bnc -> bn", a))

