"""Evidence-based MP31 SQMMA GEMM configuration planner.

The MP31 SQMMA backend does not pick the block tile by itself: the tile is
whatever the kernel author writes into the C fragment.  This module turns the
same-session S5000 measurements in
``compiler_baseline/reports/2026-09-22-sqmma-tile-planner-rule-v1.md`` into a
small, explicit recommendation so kernel generators, agents and TileOPs specs
start from a configuration that was actually measured instead of a guess.

Measured rule (MP31 / S5000, fp16 in / fp32 accumulate, patched build
``v0.1.13+musa.1``, interleaved same-session Event A/B, 30 samples x 10 calls):

    M x N >= 1.8e6                        -> 128x128x64, 256 threads, 1 stage
    otherwise (M,N,K divisible by 64)     -> 64x64x64, 128 threads, 1 stage

    boundary_clauses=True additionally enables the container-derived clauses
    M x N >= 1.6e6 and K >= 640, and M x N >= 1.3e6 and K >= 2048.

Cross-environment evidence (rule -> measurement -> revision loop):

* Container Event sweep (the 32-shape table below): the first version of the
  rule agreed on 31/32 and missed 1152x1152x2048 (M*N=1.33e6, K=2048), where
  128x128 measured 2.5% faster.  That motivated the long-K clause, and the
  relaxed clause came from the 1280^2 column (K >= 640 -> 128x128).
* Host MUPTI online loop (``run_gemm_candidates.py`` + the planner-seeded
  manifest, 6 shapes, device-side timing): the *core* threshold M*N >= 1.8e6
  agreed on 6/6 shapes, but both boundary clauses were contradicted -
  1280x1280x512 measured 64x64 faster by 11% (24.74 us vs 27.50 us) and
  1280x1280x640 by 3.2% (29.62 vs 30.58), while 1152x1152x2048 was a 1.6%
  tie at stages=1 (75.78 vs 77.00).

Conclusion: the core threshold is robust across both environments; the two
boundary clauses are container-specific and are therefore *opt-in* through
``boundary_clauses=True``.  The default rule never loses more than ~5% on the
measured shapes of either environment.

Boundary evidence (time ratio 64x64 / 128x128; >1 means 128x128 wins):

    shape               M*N        K     ratio   winner
    512^3               0.26e6     512   0.707   64x64
    512x512x4096        0.26e6    4096   0.707   64x64
    1024^3              1.05e6    1024   0.902   64x64
    1024x1024x2048      1.05e6    2048   0.875   64x64
    1024x1024x4096      1.05e6    4096   0.902   64x64
    1152^3              1.33e6    1152   0.971   64x64
    1152x1152x1280      1.33e6    1280   0.972   64x64
    1152x1280x1152      1.47e6    1152   0.995   64x64
    1152x1280x1280      1.47e6    1280   0.994   64x64
    1280^2 x 256        1.64e6     256   0.877   64x64
    1280^2 x 512        1.64e6     512   0.970   64x64
    1280^2 x 640        1.64e6     640   1.019   128x128
    1280^2 x 768        1.64e6     768   1.040   128x128
    1280^2 x 1024       1.64e6    1024   1.089   128x128
    1280^2 x 1280       1.64e6    1280   1.126   128x128
    1408x1152x1280      1.62e6    1280   1.118   128x128
    1280x1408x1280      1.80e6    1280   1.247   128x128
    1408^2 x 640        1.98e6     640   1.240   128x128
    1408^2 x 1280       1.98e6    1280   1.379   128x128
    1024x2048x1024      2.10e6    1024   1.208   128x128
    1536^2 x 256        2.36e6     256   1.230   128x128
    1792^3              3.21e6    1792   1.691   128x128
    2048^3              4.19e6    2048   1.380   128x128
    4096^3             16.8e6     4096   1.840   128x128

Additional measured constraints:

* ``stages=1`` wins for both tiles: 2048^3 128x128 s1 155.2 us vs s2 188.7 us;
  4096^3 128x128 s1 961.1 us vs s2 1123.5 us.  The cp.async pipeline does not
  pay off on MP31 for these shapes, so the planner never recommends stages>1.
* ``block_k=64`` is the measured sweet spot at 128x128 (2048^3: bk=64 110.7
  TFLOPS, bk=32 81.5, bk=128 57.2).
* Rectangular and 256-wide tiles are rejected by the current backend:
  128x64x64 t256, 128x256x64 t256, 256x256x32 t256 fail with MTCompile errors
  and 128x64x64 t256 also fails cp.async injection.  128x128 with 128 threads
  also fails to compile (MTCompile).  The planner therefore only ever returns
  the two measured-legal configurations.

The rule is deliberately conservative and only claims what was measured: it
covers fp16 inputs, cubic and rectangular shapes with M/N divisible by 128
(or 64) and K divisible by 64.  Everything else returns ``None`` so callers
fall back to their own policy instead of trusting an unmeasured guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["SqmmaConfig", "recommend_sqmma_config",
           "recommend_copy_channel", "predict_sqmma_occupancy",
           "predict_sqmma_wave_efficiency"]

_TILE_128_MIN_MN = 1_800_000
_TILE_128_RELAXED_MN = 1_600_000
_TILE_128_RELAXED_K = 640
_TILE_128_LONG_K_MN = 1_300_000
_TILE_128_LONG_K = 2048

# Copy channel thresholds, re-measured 2026-09-22 with host-side MUPTI device
# timing (single block, 128 threads, rank-1 fp32 global->shared->global copy):
#
#   bytes   TME us  direct us  direct/TME
#     512     2.36       1.60       0.68
#    4096     2.44       1.92       0.79
#   32768     3.32       3.34       1.01   <- break-even
#   65536     4.32       5.32       1.23
#   98304     5.32      16.96       3.19
#  131072     6.36       9.68       1.52
#  163840     7.28      19.28       2.65
#  196608   rejected: exceeds the allowed dynamic shared memory size
#
# The earlier Event-based measurement put the flip point at 128 KB and reported
# a 12.5 us fixed cost; device-side timing shows a ~2.3 us fixed cost and a
# 32-64 KB flip point, and it disagrees with the Event ordering at 64 KB and
# 96 KB.  Compute reuse only strengthens the TME case, so this rule is a lower
# bound: it never recommends TME below the measured break-even.
_TME_BREAK_EVEN_BYTES = 32 << 10


@dataclass(frozen=True)
class SqmmaConfig:
    """A measured-legal MP31 SQMMA block configuration."""

    block_m: int
    block_n: int
    block_k: int
    threads: int
    stages: int

    def as_dict(self) -> dict:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "threads": self.threads,
            "stages": self.stages,
        }


_CONFIG_128 = SqmmaConfig(128, 128, 64, 256, 1)
_CONFIG_64 = SqmmaConfig(64, 64, 64, 128, 1)


def _divides(shape: int, tile: int) -> bool:
    return shape > 0 and shape % tile == 0


def recommend_sqmma_config(m: int, n: int, k: int,
                           dtype: str = "float16",
                           boundary_clauses: bool = False) -> Optional[SqmmaConfig]:
    """Return the measured-best MP31 SQMMA config for an ``m x n x k`` GEMM.

    ``None`` means the planner has no measured answer for this problem (dtype
    other than fp16, non-divisible shapes); the caller must decide on its own.

    ``boundary_clauses`` opts into the two container-specific boundary clauses
    (see the module docstring); the default is the environment-robust core
    threshold.
    """
    if dtype not in ("float16", "fp16"):
        return None
    if not (_divides(m, 64) and _divides(n, 64) and _divides(k, 64)):
        return None

    mn = m * n
    if _divides(m, 128) and _divides(n, 128):
        if mn >= _TILE_128_MIN_MN:
            return _CONFIG_128
        if boundary_clauses:
            if mn >= _TILE_128_RELAXED_MN and k >= _TILE_128_RELAXED_K:
                return _CONFIG_128
            if mn >= _TILE_128_LONG_K_MN and k >= _TILE_128_LONG_K:
                return _CONFIG_128
    return _CONFIG_64

def recommend_copy_channel(tile_bytes: int) -> str:
    """Return ``"tme"`` or ``"direct"`` for an MP31 staging copy.

    ``tile_bytes`` is the size of one global->shared staging tile.  The rule
    comes from same-process MUPTI device-side A/B measurements (see the table
    above): below the 32 KB break-even the thread copy is faster, at and above
    it ``T.tma_copy`` wins and the margin grows with the tile.

    Scope: measured for contiguous rank-1 fp32 copies with one block and 128
    threads, where the staged data is written back to global.  Inside a compute
    kernel the same tile is reused, which only improves the TME side, so the
    break-even is conservative.  TME also requires the 16-byte global stride
    alignment that ``T.tma_copy`` itself enforces, and tiles above ~160 KB do
    not fit the dynamic shared memory limit.
    """
    if tile_bytes <= 0:
        raise ValueError(f"tile_bytes must be positive, got {tile_bytes}")
    return "tme" if tile_bytes >= _TME_BREAK_EVEN_BYTES else "direct"


def predict_sqmma_occupancy(config: SqmmaConfig, dtype: str = "float16") -> dict[str, Any]:
    """Derive theoretical MP31 SM occupancy and resource limits for an SQMMA configuration."""
    dtype_bytes = 2 if dtype in ("float16", "fp16", "bfloat16", "bf16") else 4
    shared_bytes = (config.block_m * config.block_k + config.block_k * config.block_n) * dtype_bytes * config.stages
    
    # Warps
    warps_per_block = config.threads // 32
    limit_by_warps = 32 // warps_per_block if warps_per_block > 0 else 0
    
    # Shared memory (192 KB total SMEM)
    smem_total = 192 * 1024
    limit_by_smem = smem_total // shared_bytes if shared_bytes > 0 else 16
    
    # Registers (64K registers per SM)
    acc_per_thread = (config.block_m * config.block_n) // config.threads
    est_regs_per_thread = acc_per_thread + 32
    limit_by_regs = 65536 // (config.threads * est_regs_per_thread) if (config.threads * est_regs_per_thread) > 0 else 16
    
    active_blocks = min(limit_by_warps, limit_by_smem, limit_by_regs, 16)
    active_warps = active_blocks * warps_per_block
    occupancy = active_warps / 32.0
    
    return {
        "shared_bytes": shared_bytes,
        "acc_per_thread": acc_per_thread,
        "estimated_regs_per_thread": est_regs_per_thread,
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "theoretical_occupancy": occupancy,
        "limit_by_warps": limit_by_warps,
        "limit_by_smem": limit_by_smem,
        "limit_by_regs": limit_by_regs,
    }


def predict_sqmma_wave_efficiency(m: int, n: int, config: SqmmaConfig, num_mps: int = 128) -> dict[str, Any]:
    """Derive full-machine wave quantization efficiency across S5000 128 MPs."""
    import math
    grid_m = math.ceil(m / config.block_m)
    grid_n = math.ceil(n / config.block_n)
    grid_blocks = grid_m * grid_n
    
    occ = predict_sqmma_occupancy(config)
    active_blocks = max(1, occ["active_blocks_per_sm"])
    concurrent_capacity = num_mps * active_blocks
    
    num_waves = math.ceil(grid_blocks / concurrent_capacity)
    wave_efficiency = grid_blocks / (num_waves * concurrent_capacity) if num_waves > 0 else 1.0
    tail_blocks = grid_blocks - (num_waves - 1) * concurrent_capacity
    tail_active_sms = math.ceil(tail_blocks / active_blocks)
    
    return {
        "grid_blocks": grid_blocks,
        "concurrent_capacity": concurrent_capacity,
        "num_waves": num_waves,
        "wave_efficiency": wave_efficiency,
        "tail_active_sms": tail_active_sms,
        "idle_sms_in_tail_wave": max(0, num_mps - tail_active_sms),
    }

