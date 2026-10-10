# PTX Analysis

All PTX was dumped from Triton on a Colab T4 (`.target sm_75`), fp32, via `compiled.asm['ptx']`. Files referenced here live in this folder.

Pipeline reminder: Triton Python to Triton IR to LLVM IR to NVPTX backend to PTX, then `ptxas` assembles PTX to SASS. PTX registers are virtual. Real register counts come from `ptxas` (`compiled.n_regs`).

---

## 1. FMA kernel (`fma_kernel.ptx`)

Source: `out = x * y + z`, `BLOCK_SIZE=1024` (constexpr), default `num_warps=4`.

| Triton source | PTX | What the compiler did |
|---|---|---|
| launch config | `.reqntid 128` | 128 threads per block (4 warps) |
| `pid * BLOCK_SIZE` | `shl.b32 %r34, %r33, 10` | Strength reduction: multiply by constexpr 1024 became a shift by 10 |
| `tl.arange` offsets | `shl 2`, `and 508`, `or 512` | Each thread owns two groups of 4 contiguous floats (8 per thread, 128 x 8 = 1024) |
| `mask = offsets < n` | `setp.lt.s32 %p1, ...` and `%p2` | Mask became predicate registers |
| `tl.load(..., mask)` | `@%p1 ld.global.v4.b32` | Vectorized 16 byte loads, predicated, no branch |
| `x * y + z` | `fma.rn.f32` (8 per thread) | Mul and add fused into one instruction per element |
| `tl.store(..., mask)` | `@%p1 st.global.v4.b32` | Vectorized, predicated store |

Findings:

1. I wrote scalar looking code. The compiler chose 4 wide vector memory ops and 8 elements per thread on its own.
2. Boundary masking is predication, not branching, so no warp divergence cost.
3. `x * y + z` became one `fma.rn.f32`, confirming fusion.
4. Constant folding and strength reduction are visible because `BLOCK_SIZE` is `tl.constexpr`.

### Experiment A: non power of 2 BLOCK_SIZE

`BLOCK_SIZE=1000` fails at compile time:

```
CompilationError ... arange's range must be a power of 2
```

`tl.arange` requires a power of 2 length. Non power of 2 problem sizes are handled by masking, not by odd block sizes.

### Experiment B: constexpr vs runtime stride (`fma_runtime_stride.ptx`)

Replaced `pid * BLOCK_SIZE` with `pid * stride`, where `stride` is a runtime argument (value 1024).

| Version | PTX for the multiply |
|---|---|
| constexpr | `shl.b32 %r34, %r33, 10` |
| runtime arg | `mad.lo.s32 %r39, %r38, %r33, %r37` |

- Compiler cannot know the value at compile time, so the shift disappears and a fused multiply-add on integers is used.
- `ld.global.v4.b32` vectorization is preserved (6 vector loads, same as before).
- Correctness: `allclose` True, max abs diff 0.0.
- The runtime version shows two `mul.wide.s32` address computations where the constexpr version showed one plus a constant `+2048`. I did not isolate why.
- No performance claim is made. The kernel is memory bound.

---

## 2. Softmax kernel (`softmax_kernel.ptx`)

Source: one program per row, whole row in one block. `n_cols=1024`, `BLOCK_SIZE=1024`, `num_warps=4` (`.reqntid 128`). `ptxas` reports 23 registers per thread and 0 spills. PTX declares `%r<107>` virtual registers, which is not the real count.

| Triton source | PTX | What the compiler did |
|---|---|---|
| `row_idx * stride` | `mul.lo.s32`, `mad.wide.s32` | Pointer arithmetic fused |
| offsets | `shl 2`, `and 508`, `or 512` | 8 elements per thread, two 4 wide groups |
| `tl.load(..., other=-inf)` | `mov.b32 %r5, -8388608` then `@%p1 ld.global.v4.b32` | `-8388608` is `0xFF800000`, fp32 `-inf`. Destination registers prefilled, predicated load overwrites valid lanes |
| `tl.max(row)` | 7 `max.f32`, then 5 `shfl.sync.bfly` (16, 8, 4, 2, 1) | In thread chain, then warp butterfly reduction |
| `tl.max` cross warp | `st.shared`, `bar.sync`, `ld.shared`, 2 more `shfl`, `bar.sync` | Lane 0 of each warp writes shared memory, 4 warp values reduced |
| `row - row_max` | 8 `sub.f32` | |
| `tl.exp` | `mul.f32 x, 0f3FB8AA3B` then `ex2.approx.f32` | `0x3FB8AA3B` is log2(e). `exp(x)` lowered to `2^(x * log2e)` on the hardware ex2 unit |
| `tl.sum` | `bar.sync`, 7 `add.f32`, 5 `shfl`, shared memory stage | Same three level hierarchy as max |
| `numerator / denominator` | 8 `div.full.f32` | One division per element |
| `tl.store(..., mask)` | `@%p1 st.global.v4.b32` | Vectorized, predicated |

Findings:

1. One fused kernel: one global read and one global write per element, with max, exp, sum, divide done in registers.
2. Reductions lower to three levels: in thread, warp shuffle, cross warp through shared memory. Five `bar.sync` in total (two for max, three for sum).
3. `tl.exp` lowers to `ex2.approx.f32`. This is an approximate instruction. Measured max abs diff vs PyTorch: 7.45e-9 on a 1823 x 781 input, inside `atol=rtol=1e-5`.
4. The extra `bar.sync` before the sum likely protects reuse of the shared memory slot. This is inferred from structure, not verified.
5. CSE visible: thread byte offset `%rd8` is computed once and reused for the input and output pointers. The second vector group is a constant `+2048` byte offset.
6. Missed opportunity: 8 `div.full.f32` instead of one reciprocal plus 8 multiplies. Compiler cannot do this rewrite safely, because last bit results differ. Not tested in source.
7. No LICM here. The kernel has no loop, so there is nothing to hoist.

---

## 3. num_warps sweep (T4, fp32, 4096 rows)

Registers and spills are deterministic across runs. Bandwidth varies about 1 to 1.5 percent run to run. Theoretical occupancy uses 65536 registers per SM, 1024 threads per SM, 32 warps per SM (approximate allocation model).

| n_cols | num_warps | n_regs | n_spills | theoretical occupancy | GB/s (one run) |
|---|---|---|---|---|---|
| 1024 | 2 | 31 | 0 | 100% | 231.7 |
| 1024 | 4 | 23 | 0 | 100% | 232.1 |
| 1024 | 8 | 20 | 0 | 100% | 235.9 |
| 1024 | 16 | 18 | 0 | 100% | 238.5 |
| 4096 | 2 | 90 | 0 | 62.5% | 225.6 |
| 4096 | 4 | 51 | 0 | 100% | 227.4 |
| 4096 | 8 | 36 | 0 | 100% | 229.0 |
| 4096 | 16 | 25 | 0 | 100% | 232.0 |
| 8192 | 2 | 168 | 0 | 37.5% | 225.3 |
| 8192 | 4 | 90 | 0 | 62.5% | 227.2 |
| 8192 | 8 | 52 | 0 | 100% | 228.8 |
| 8192 | 16 | 35 | 0 | 100% | 230.3 |

Registers scale with elements per thread because the whole row is held in registers. At `n_cols=8192`, 168 registers at 2 warps falls to 35 at 16 warps.

---

## 4. PyTorch softmax launch config (profiler)

From `torch.profiler` chrome traces. Launch config only. The PyTorch kernels are precompiled CUDA, so no PTX is available here. `dur_us` from the profiler is inflated by overhead and is not used for timing.

| n_cols | kernel | grid | block | regs | smem | est. occupancy |
|---|---|---|---|---|---|---|
| 1024 | `softmax_warp_forward` | 1024 | 32 x 4 (128 threads) | 64 | 0 | 100% |
| 4096 | `cunn_SoftMaxForwardReg` | 4096 | 1024 | 29 | 128 B | 100% |
| 8192 | `cunn_SoftMaxForwardReg` | 4096 | 1024 | 35 | 128 B | 100% |

PyTorch switches strategy by size: roughly warp per row at 1024 (shuffles only, no shared memory), block per row with 1024 threads at 4096 and above. My Triton kernel uses one strategy at all sizes, and my sweep stopped at 16 warps (512 threads).
