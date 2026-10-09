import torch
import triton
import os
import pandas as pd

from kernels.softmax_kernel import softmax_triton
from benchmarks.utils import save_ptx

def run_benchmark(n_rows=4096, sizes=(128, 256, 512, 1024, 2048, 4096, 8192)):
    rows = []
    for n_cols in sizes:
        x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)

        ms_torch = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
        ms_triton = triton.testing.do_bench(lambda: softmax_triton(x))

        gb = 2 * x.numel() * x.element_size() / 1e9
        rows.append({
            'n_cols': n_cols,
            'torch_ms': ms_torch,
            'triton_ms': ms_triton,
            'torch_gbps': gb / (ms_torch * 1e-3),
            'triton_gbps': gb / (ms_triton * 1e-3),
            'speedup': ms_torch / ms_triton,
        })
        print(f"n_cols={n_cols:5d} torch={ms_torch:.4f}ms triton={ms_triton:.4f}ms "
              f"speedup={ms_torch / ms_triton:.2f}x")

    df = pd.DataFrame(rows)
    os.makedirs('results/plots', exist_ok=True)
    df.to_csv('results/softmax_benchmark.csv', index=False)
    return df


def plot(df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(df['n_cols'], df['torch_gbps'], marker='o', label='PyTorch')
    plt.plot(df['n_cols'], df['triton_gbps'], marker='s', label='Triton')
    plt.axhline(320, color='gray', linestyle='--', label='T4 peak (~320 GB/s)')
    plt.xscale('log', base=2)
    plt.xlabel('n_cols')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Softmax bandwidth: Triton vs PyTorch (T4)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/softmax_bandwidth_vs_size.png', dpi=150)

def check_correctness(n_rows=4096, n_cols=1024):
    x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)

    out_torch = torch.softmax(x, dim=-1)
    out_triton, _ = softmax_triton(x, return_compiled=True)

    diff = (out_triton - out_torch).abs()
    print(f"Max absolute difference: {diff.max().item()}")
    print(f"Mean absolute difference: {diff.mean().item()}")

    if not torch.allclose(out_torch, out_triton, atol=1e-6):
        raise ValueError("Triton softmax output does not match PyTorch output!")

    print("Correctness check passed: outputs match.")
    save_ptx(_, 'softmax_kernel.ptx')

def warp_tuning(n_rows=4096):
    rows = []
    for n_cols in [1024, 4096, 8192]:
        x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
        gb = 2 * x.numel() * x.element_size() / 1e9
        for nw in [2, 4, 8, 16]:
            _, compiled = softmax_triton(x, num_warps=nw, return_compiled=True)
            ms = triton.testing.do_bench(lambda: softmax_triton(x, num_warps=nw))
            rows.append({
                'n_cols': n_cols,
                'num_warps': nw,
                'n_regs': compiled.n_regs,
                'n_spills': compiled.n_spills,
                'ms': ms,
                'gbps': gb / (ms * 1e-3),
            })
            print(rows[-1])

    pd.DataFrame(rows).to_csv('results/softmax_warp_tuning.csv', index=False)

if __name__ == '__main__':
    check_correctness()
    df = run_benchmark()
    warp_tuning()
    plot(df)