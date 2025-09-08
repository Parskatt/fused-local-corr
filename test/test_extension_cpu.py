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


device = "cpu"
num_batch, hw, height, width, channels, num_corr = 2, 64*64, 64, 64, 1024, 64
a = torch.randn(num_batch, hw, channels).to(device)
b = torch.randn(num_batch, height, width, channels).to(device)
cols = torch.randint(0, width, (num_batch, hw, num_corr)).to(device)
rows = torch.randint(0, height, (num_batch, hw, num_corr)).to(device)
c = torch.stack((cols, rows), dim = -1).int()

T = 10
import time
def perf(f, name):
    f(a,b,c)
    t0 = time.perf_counter()
    for t in range(T):
        f(a,b,c)
    print(f"Time: {(time.perf_counter()-t0)/T} for {name}")

# naive_compiled = torch.compile(naive_corr)

perf(baseline_corr,"baseline_corr")
# perf(naive_compiled,"naive_corr_compiled")
perf(corr,"custom_corr")


# print(torch.einsum("bnc -> bn", a))

