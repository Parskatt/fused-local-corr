from local_corr import corr
import torch

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

T = 10
import time


device = "cpu"
B, HW, H, W, C, N = 8, 32*32, 32, 32, 512, 64
im_A = torch.randn(B, HW, C).to(device).requires_grad_()
im_B = torch.randn(B, H, W, C).to(device)
cols = torch.randint(0, W, (B, HW, N)).to(device)
rows = torch.randint(0, H, (B, HW, N)).to(device)
warp = torch.stack((cols, rows), dim = -1).int().float()


def perf(f, name):
    torch.cuda.reset_max_memory_allocated()
    res = f(im_A, im_B, warp).mean()
    res.backward()
    t0 = time.perf_counter()
    for t in range(T):
        res = f(im_A, im_B, warp).mean()
        res.backward()
        # torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated()/10**6, "MB")
    torch.cuda.reset_max_memory_allocated()

    print(f"Time: {(time.perf_counter()-t0)/T} for {name}")

perf(corr,"custom_corr")
perf(baseline_corr,"naive_corr")
# perf(torch.compile(corr),"custom_corr_compiled")

