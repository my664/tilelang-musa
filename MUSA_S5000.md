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
