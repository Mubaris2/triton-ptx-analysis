import torch
import triton


def run_benchmark():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    n_rows = 4096
    results = []

    for n_cols in sizes:
        x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
        gb = 2 * x.numel() * x.element_size() / 1e9
        gbps = gb / (ms * 1e-3)
        results.append((n_cols, ms, gbps))
        print(f"n_cols={n_cols}: latency={ms:.4f}ms bandwidth={gbps:.2f}GB/s")

    return results


if __name__ == '__main__':
    run_benchmark()