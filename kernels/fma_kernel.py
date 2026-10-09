import torch
import triton
import triton.language as tl

@triton.jit
def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    out = x * y + z
    tl.store(out_ptr + offsets, out, mask=mask)

def fma_triton(x, y, z, block_size=1024, num_warps=4, return_compiled=False):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    compiled = fma_kernel[grid](x, y, z, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    if return_compiled:
        return out, compiled
    return out