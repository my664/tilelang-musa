# MUSA MP31 / MTT S5000 adaptation patch stack

This branch (`s5000/mp31-adaptation-v1`) carries the compiler-side work needed to
run and tune TileLang on Moore Threads MP31 (MTT S5000), on top of
`489df49a Prepare public`.  Every change below was validated on real S5000
hardware with same-session A/B measurements; the evidence (reports, raw JSON,
probe scripts, matrices) lives in the companion repository
[`my664/tilelang-musa-work`](https://github.com/my664/tilelang-musa-work).

## Commits

| Commit | Area | What it fixes / adds |
| --- | --- | --- |
| `musa: add source-only build mode` | build | `USE_MUSA=OFF` source-only build (lowering/codegen without the device compiler) |
| `musa: fix thread storage sync byte-range disjointness` | pass | byte-range disjointness in `thread_storage_sync.cc`; removes spurious or missing barriers |
| `musa: add warp-first AllReduce templates` | templates | `WarpFirstAllReduce` / batch AllReduce for MUSA |
| `musa: add source-only lowering tests` | tests | source-only lowering tests for copy/rmsnorm/cumsum |
| `musa: invalidate SSA cache on extern calls` | codegen | fixes cumsum in-place write dependency (max rel error 2.1e7 -> 1.4e-6) |
| `musa: keep the TVM SSA-invalidation companion patch` | submodule | TVM side of the same fix, kept as `musa_patches/tvm-codegen-ssa-invalidate.patch` |
| `musa: let T.gemm auto-select MP31 SQMMA` | gemm | plain `T.gemm` lowers to MP31 SQMMA instead of failing |
| `musa: pipelined SQMMA shared layout` | layout | SQMMA shared layout usable inside `T.Pipelined` |
| `musa: async pipeline copy support` | copy | pipeline-managed `cp.async` copies |
| `musa: cp.async commit/wait group protocol` | templates | commit/wait group protocol for the MUSA async copy path |
| `musa: evidence-based SQMMA config planner (MP31/S5000)` | planner | `recommend_sqmma_config()` from measured tile rules |
| `musa: guard SQMMA register budget with actionable diagnostics` | gemm/copy | replaces a cryptic `MTCompile` register-allocation crash with an actionable error; better async-copy diagnostics |
| `musa: revise planner rule from the rule -> measurement loop; add copy channel rule` | planner | long-K clause from the 32-shape validation loop; `recommend_copy_channel()` from MUPTI measurements |

## GEMM / TME stack (2026-10-09 -> 2026-10-20)

The second wave closes the gap between "the pipeline is configured" and "the
pipeline actually runs". Ordered as the dependency chain, not chronologically.

| Commit | Area | What it fixes / adds |
| --- | --- | --- |
| `musa: eliminate redundant shared staging` (+ completion fix) | pass | `EliminateRedundantSharedStaging` drops the extra shared-memory hop in front of `T.gemm`; finishes `IfThenElse` unwrapping and `SBlock` alloc cleanup so the pass fires on real kernels |
| `musa: add analytical MP31 occupancy and wave efficiency prediction helpers` | planner | `predict_sqmma_occupancy()` / `predict_sqmma_wave_efficiency()` give the planner closed-form occupancy and wave quantisation instead of measured tables |
| `musa: gate MP31 cp.async pipeline explicitly` | pipeline | `mp31_pipeline_async` attribute gates `tirx.use_async_copy` (opt-in first, still synchronous by default) |
| `fix(musa): enable async copy injection on MP31` | pass | **root cause of the "async configured but serial in practice" bug**: `LowerTileOp` tested `TargetCudaHasAsyncCopy()` and skipped MUSA entirely; replaced by `TargetHasAsyncCopy()` |
| `musa: auto-tile MP31 GEMM and lower rotating TME operands` | gemm | automatic tile selection for `T.gemm`, plus rotating (ring-buffer) TME operand lowering; introduces `tme_layout.py` with the 256-byte XOR layout contract check |
| `fix(musa): route auto sqmma through explicit backend` | gemm | auto-selected SQMMA goes through the explicit backend so the planner config is honoured rather than defaulted |
| `musa: enable TME hardware async copy lowering and barrier binding` | copy | Binds `tme_load` to `tl.mbarrier_arrive` / `tl.mbarrier_wait_parity` instead of lowering to a thread copy |
| `musa: support dual-squad 256x128 SQMMA tiling with partitioned accumulators` | gemm | 256x128 tile spread over two squads (>=256 threads); accumulators partitioned so each thread stays within the 64 fp32 register budget |
| `musa: reduce redundant barriers on async completion and enable threadblock swizzle` | codegen/pass | Removes the completion-time barrier that serialised the producer, adds `threadblock_swizzle.h` for L2-friendly block ordering |
| `musa: native hardware TME swizzle lowering and remove software layout conversion` | gemm/codegen/templates | `tme_layout.py` now annotates each `tl.tileop.tma_copy` with `swizzle_granularity` (16/32/64 B taken from the SQMMA shared layout, validated against the 256-byte XOR contract); `copy.cc` maps 16 -> `__musa::SG_16B` and 32 -> `__musa::SG_32B`; `codegen_musa.cc` emits the enum; `tme.h` takes `int sg` on the 1D-5D `tme_load`. **The software `tme_sqmma_swizzle_inplace` permutation and its two global `__syncthreads()` are removed** - the layout is now done by the TME descriptor, not by a memory shuffle |
| `fix: align planner copy channel and accumulator budget with test contract` | planner/gemm | Normalises a recommended `"direct"` copy channel to `"sync"` (what `make_sqmma_gemm` actually validates) and aligns the accumulator budget with the dual-squad contract |

## Measured outcomes (S5000, MUSA 5.2.0)

* operator matrix (29 specs, device): 29/29 pass, no regressions
* rmsnorm 1.28x, softmax 1.17x vs the same build without the stack (same-session A/B)
* cumsum in-place write dependency: max relative error 2.1e7 -> 1.4e-6
* SQMMA tile planner: 1536^3 1.81x, 2048^3 1.38x (110.7 TFLOPS), 4096^3 1.84x
  versus the 64x64 tile
* TME copy channel (device-side MUPTI): break-even at 32 KB, 96 KB 3.19x,
  160 KB 2.65x versus the thread copy
* MP31 SQMMA legality: tiles needing >64 fp32 accumulators per thread on
  large native shapes are refused with a clear message instead of crashing
  the device compiler

### GEMM 2048x4096x4096 FP16 (tile 128x128x64, 256 threads, TME, 2 stages)

Device receipts in `tilelang-musa-work`, `compiler_baseline/results/`:

| Run | Layout path | Median | TFLOPS | vs muBLAS | Correctness |
| --- | --- | --- | --- | --- | --- |
| `musagen_s5000_e2e_20261016_run9` | software `tme_sqmma_swizzle_inplace` | 0.766 ms | 89.71 | 29.46% | `max_abs_error` 0.0 |
| `musagen_s5000_e2e_20261020_run10` | hardware TME swizzle (`SG_16B/32B`) | 0.3938 ms | 174.50 | 55.64% | `max_abs_error` 0.0 |
| `musagen_evolution` best (G6) | run10 + agent-written epilogue | 0.3832 ms | 179.31 | 58.67% | `max_abs_error` 4.88e-4 |

Reading notes:

* **Compare TFLOPS, not percentages.** muBLAS itself measured between 239 and
  314 TFLOPS across these sessions (shared device, clock variation), so the
  ratio moves even when our kernel is unchanged.
* The 89.71 -> 174.50 step is the compiler stack (hardware swizzle lowering);
  the agent contribution on top is +4.81 TFLOPS over 7 generations, of which
  only generations 2, 5 and 6 improved (1, 3, 4 and 7 regressed and were kept
  out of the record).
* The project target for this shape is >=85% of muBLAS. Current best is
  ~58%, i.e. the remaining gap is performance engineering, not correctness or
  missing lowering.

## Building

Source-only (no device compiler, works on any host):

```bash
cmake -S . -B build -DUSE_MUSA=OFF && cmake --build build -j
PYTHONPATH=$PWD LD_LIBRARY_PATH=$PWD/build/lib python -c "import tilelang; print(tilelang.__version__)"
```

Full build for S5000 needs MUSA SDK 5.2.0 (`mcc`) and the MUSA torch build; see
`compiler_baseline/matrix/env_host_s5000.sh` in the companion repository.

## TVM submodule

`3rdparty/tvm` needs the companion patch before the SSA fix is complete:

```bash
git -C 3rdparty/tvm apply musa_patches/tvm-codegen-ssa-invalidate.patch
```

## Opt-in / rejected candidates

Two patches are deliberately **not** part of this branch and live in the
companion repository:

* `musa-reduce-pack-v1`: packed x2 reduction. Implemented and correct, but
  device-side timing showed zero benefit on the measured workloads (the pack
  path only vectorizes the local accumulation and rarely activates), so it is
  kept as an opt-in patch.
* `r1-lower-thread-allreduce-musa-candidate.patch`: allows MUSA in the shared
  warp-reduction pass. Found inert for TileLang reduce ops (they lower to
  `tl::AllReduce` templates directly), kept as a candidate.

## Target warp-size normalization (local follow-up)

The working tree also contains the fail-closed MP21/MP22/MP31 warp-size normalization in 	ilelang/musa/target.py and 	esting/python/target/test_tilelang_musa_target.py. The matching TVM target-kind patch is preserved at musa_patches/tvm-target-warp-size.patch; it must be applied inside 3rdparty/tvm before validating explicit MP21/MP22 targets. This follow-up has source-level regression coverage but no MP21/MP22 device result yet.
