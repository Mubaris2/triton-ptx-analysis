import torch
import triton
import os
import pandas as pd

from kernels.fma_kernel import fma_triton
from benchmarks.utils import save_ptx

def run_benchmark(sizes=(2**14, 2**16, 2**18, 2**20, 2**22, 2**24, 2**26)):
    rows = []
    for n in sizes:
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        z = torch.randn(n, device='cuda')

        ms_torch = triton.testing.do_bench(lambda: x * y + z)
        ms_triton = triton.testing.do_bench(lambda: fma_triton(x, y, z))

        gb = 4 * n * x.element_size() / 1e9
        rows.append({
            'n_elements': n,
            'torch_ms': ms_torch,
            'triton_ms': ms_triton,
            'torch_gbps': gb / (ms_torch * 1e-3),
            'triton_gbps': gb / (ms_triton * 1e-3),
            'speedup': ms_torch / ms_triton,
        })
        print(f"n={n:10d} torch={ms_torch:.4f}ms triton={ms_triton:.4f}ms "
              f"speedup={ms_torch / ms_triton:.2f}x")

    df = pd.DataFrame(rows)
    os.makedirs('results/plots', exist_ok=True)
    df.to_csv('results/fma_benchmark.csv', index=False)
    return df

def plot(df):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df['n_elements'], df['torch_gbps'], marker='o', label='PyTorch (x*y+z)')
    ax1.plot(df['n_elements'], df['triton_gbps'], marker='s', label='Triton fma_kernel')
    ax1.axhline(320, color='gray', linestyle='--', label='T4 peak (~320 GB/s)')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('n_elements (fp32)')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('FMA bandwidth (T4)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(df)), df['speedup'])
    ax2.axhline(1.0, color='red', linestyle='--')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f"2^{int(n).bit_length() - 1}" for n in df['n_elements']])
    ax2.set_xlabel('n_elements')
    ax2.set_ylabel('Speedup (PyTorch ms / Triton ms)')
    ax2.set_title('Speedup (above 1.0 = Triton faster)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/plots/fma_benchmark.png', dpi=150)

def check_correctness(n=2**20):
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    z = torch.randn(n, device='cuda')

    out_torch = x * y + z
    out_triton, _ = fma_triton(x, y, z, return_compiled=True)

    diff = (out_triton - out_torch).abs()
    print(f"Max absolute difference: {diff.max().item()}")
    print(f"Mean absolute difference: {diff.mean().item()}")

    if not torch.allclose(out_torch, out_triton, atol=1e-6):
        raise ValueError("Triton fma_kernel output does not match PyTorch output!")

    print("Correctness check passed: outputs match.")
    save_ptx(_, 'fma_kernel.ptx')

if __name__ == '__main__':
    check_correctness()
    df = run_benchmark()
    plot(df)