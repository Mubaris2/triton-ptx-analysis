import torch
import triton
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kernels.fma_kernel import fma_triton


def run_benchmark():
    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    for n in sizes:
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        z = torch.randn(n, device='cuda')

        ms_triton = triton.testing.do_bench(lambda: fma_triton(x, y, z))
        ms_torch = triton.testing.do_bench(lambda: x * y + z)

        results.append({
            'n_elements': n,
            'triton_ms': ms_triton,
            'torch_ms': ms_torch,
            'speedup': ms_torch / ms_triton
        })
        print(f"n={n}: triton={ms_triton:.4f}ms torch={ms_torch:.4f}ms speedup={ms_torch/ms_triton:.2f}x")

    return results


if __name__ == '__main__':
    run_benchmark()