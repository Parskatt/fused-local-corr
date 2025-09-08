from local_corr import corr
import torch
import torch.nn.functional as F

def baseline_corr(im_A: torch.Tensor, im_B: torch.Tensor, warp: torch.Tensor):
    B, H, W, C = im_B.shape
    inds = warp[..., 1]*W + warp[..., 0]
    im_B_warped = torch.take_along_dim(
        im_B.reshape(B, 1, H*W, C),
        inds[..., None].long(),
        dim=2,
    )
    out = torch.einsum("bnc, bnmc -> bnm", im_A, im_B_warped)
    return out

device = "cuda"
B, HW, H, W, C, N = 8, 64*64, 64, 64, 64, 64*64
im_A = torch.randn(B, HW, C).to(device)
im_B = torch.randn(B, H, W, C).to(device)
cols = torch.randint(0, W, (B, HW, N)).to(device)
rows = torch.randint(0, H, (B, HW, N)).to(device)
warp = torch.stack((cols, rows), dim = -1).int()


T = 10
import time

def perf(f, name):
    with torch.inference_mode():
        f(im_A,im_B,warp)
        t0 = time.perf_counter()
        for t in range(T):
            f(im_A,im_B,warp)
        print(f"Time: {(time.perf_counter()-t0)/T} for {name}")

naive_compiled = torch.compile(baseline_corr)
# perf(baseline_corr, "baseline")
perf(corr, "custom")

# perf(new_corr,"naive_corr")
# perf(naive_compiled,"naive_corr_compiled")
# perf(corr,"custom_corr")


# print(torch.einsum("bnc -> bn", a))

