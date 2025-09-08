from local_corr import corr
import torch
import torch.nn.functional as F

def baseline_corr_bilinear(im_A: torch.Tensor, im_B: torch.Tensor, warp: torch.Tensor):
    B, H, W, C = im_B.shape
    im_B_warped = F.grid_sample(
        im_B.permute(0,3,1,2),
        warp,
        mode = "bilinear",
        align_corners=False,
    )
    out = torch.einsum("bnc, bcnm -> bnm", im_A, im_B_warped)
    return out


device = "cuda"
B, HW, H, W, C, N = 8, 64*64, 64, 64, 1024, 64
im_A = torch.randn(B, HW, C).to(device)
im_B = torch.randn(B, H, W, C).to(device)
warp = (torch.rand((B, HW, N, 2)).to(device)-.5)*2

T = 10
import time

def perf(f, name):
    with torch.inference_mode():
        f(im_A,im_B,warp)
        t0 = time.perf_counter()
        for t in range(T):
            f(im_A,im_B,warp)
        print(f"Time: {(time.perf_counter()-t0)/T} for {name}")

torch.set_float32_matmul_precision('highest')

im_A.requires_grad = True
naive_compiled = torch.compile(baseline_corr_bilinear)
perf(baseline_corr_bilinear, "baseline_bilinear")
perf(naive_compiled, "baseline_compiled_bilinear")
