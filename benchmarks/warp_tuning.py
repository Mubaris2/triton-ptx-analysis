import torch
import triton
import os
import pandas as pd

from kernels.softmax_kernel import softmax_triton   

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

    df = pd.DataFrame(rows)
    os.makedirs('results/plots', exist_ok=True)
    df.to_csv('results/softmax_warp_tuning.csv', index=False)
    return df

def plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (n_cols, g) in zip(axes, df.groupby('n_cols')):
        g = g.sort_values('num_warps')
        ax.plot(g['num_warps'], g['gbps'], marker='o')
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['n_regs'])} regs",
                        (r['num_warps'], r['gbps']),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8)
        ax.axhline(240, color='gray', linestyle='--', label='clone ceiling (~240 GB/s)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(g['num_warps'])
        ax.set_xticklabels(g['num_warps'])
        ax.set_xlabel('num_warps')
        ax.set_title(f'n_cols={n_cols}')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Bandwidth (GB/s)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Triton softmax: num_warps vs bandwidth (T4, fp32, spills=0 everywhere)')
    plt.tight_layout()
    plt.savefig('results/plots/softmax_num_warps_tuning.png', dpi=150)

if __name__ == '__main__':
    df = warp_tuning()
    plot(df)