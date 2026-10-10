# Findings

Setup: Colab Tesla T4 (sm_75), fp32, Triton with the NVPTX backend. Timing via `triton.testing.do_bench`. Softmax uses 4096 rows. Spec peak bandwidth is about 320 GB/s, but a plain `x.clone()` (one read, one write, same bytes) reaches only about 240 GB/s, so 240 GB/s is the practical ceiling used below.

## 1. Practical bandwidth ceiling

| n_cols | `x.clone()` GB/s |
|---|---|
| 1024 | 235.9 |
| 4096 | 241.0 |
| 8192 | 240.9 |

Softmax and FMA on this GPU are memory bound. Both PyTorch and Triton land within a few percent of this ceiling.

## 2. Softmax: Triton vs PyTorch

Same script, same session. Dip at 4096 and above reproduced on rerun.

| n_cols | PyTorch ms | Triton ms (num_warps=4) | speedup |
|---|---|---|---|
| 128 | 0.0233 | 0.0412 | 0.57x |
| 256 | 0.0399 | 0.0386 | 1.03x |
| 512 | 0.0778 | 0.0761 | 1.02x |
| 1024 | 0.1580 | 0.1478 | 1.07x |
| 2048 | 0.3156 | 0.2934 | 1.08x |
| 4096 | 0.5526 | 0.5884 | 0.94x |
| 8192 | 1.1157 | 1.1777 | 0.95x |

Reading:

- Mid sizes: parity, with Triton up to about 1.08x. Both kernels are fused and memory bound, so little room exists.
- n_cols=128: Triton slower. At about 20 microseconds the Python wrapper and launch path likely dominate. Not isolated.
- n_cols of 4096 and above: Triton about 5 percent slower at default `num_warps=4`.
- Best `num_warps` per size narrows the gap but does not close it. At 8192, best Triton is about 1.165 to 1.170 ms versus PyTorch 1.1157 ms (roughly 0.95 to 0.96x).
- Speedups under about 2 percent are inside run to run noise.

## 3. Hypothesis tested and rejected: register pressure

Hypothesis: the whole row sits in registers, so at large rows registers per thread rise, occupancy falls, and Triton slows down.

Evidence against it:

- `n_spills` was 0 in all 12 configurations.
- At n_cols=8192, registers per thread ranged from 168 to 35 and theoretical occupancy from 37.5 percent to 100 percent across `num_warps` 2 to 16.
- Bandwidth moved only about 2 percent (roughly 225 to 230 GB/s) across that whole range.

Conclusion: occupancy nearly tripled and bandwidth barely moved. The kernel is memory bound, not register bound. Occupancy numbers are theoretical estimates, not Nsight measurements.

## 4. Remaining gap to PyTorch at large rows: unexplained

Best Triton at 4096 and 8192 sits about 3 to 4 percent under the `clone` ceiling. Profiling PyTorch shows a different strategy:

- 1024 columns: warp per row, 128 threads, no shared memory.
- 4096 and above: block per row, 1024 threads, 128 B shared memory (consistent with one float per warp).

My sweep stopped at 16 warps and bandwidth was still rising at large sizes, with no plateau. Using more warps may close the gap, but this was not tested, so it stays a hypothesis.

## 5. What the compiler did (see `ptx_analysis/analysis.md`)

- Auto vectorization: scalar looking Triton became `ld.global.v4` and `st.global.v4`, 8 elements per thread.
- Predication instead of branching for boundary masks, so no warp divergence cost.
- `x * y + z` fused into `fma.rn.f32`.
- Strength reduction: `pid * BLOCK_SIZE` became `shl` when `BLOCK_SIZE` is `tl.constexpr`, and became `mad.lo.s32` when the stride is a runtime argument.
- `tl.exp` lowered to `mul` by log2(e) plus `ex2.approx.f32`.
- Reductions lowered to a three level hierarchy: in thread `max`/`add` chain, warp `shfl.sync.bfly`, cross warp through shared memory with `bar.sync`.
- Compiler left 8 `div.full.f32` instead of one reciprocal and multiplies. Not tested as a source level fix.
- `tl.arange` rejects non power of 2 sizes at compile time.
- PTX declares 107 virtual 32 bit registers for softmax, but `ptxas` allocates 23 real registers.

## 6. Limitations

- Single GPU (T4), fp32 only, one row count.
- Occupancy is a theoretical calculation. Nsight Compute was not used.
- PyTorch kernels were inspected through launch config only. No PyTorch PTX or SASS was compared.
- LICM was not observed, because neither kernel contains a loop.
- Cause of the 3 to 4 percent gap at large rows is not established.
