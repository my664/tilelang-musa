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
    M x N >= 1.6e6 and K >= 640           -> 128x128x64, 256 threads, 1 stage
    otherwise (M,N,K divisible by 64)     -> 64x64x64, 128 threads, 1 stage

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
from typing import Optional

__all__ = ["SqmmaConfig", "recommend_sqmma_config"]

_TILE_128_MIN_MN = 1_800_000
_TILE_128_RELAXED_MN = 1_600_000
_TILE_128_RELAXED_K = 640


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
                           dtype: str = "float16") -> Optional[SqmmaConfig]:
    """Return the measured-best MP31 SQMMA config for an ``m x n x k`` GEMM.

    ``None`` means the planner has no measured answer for this problem (dtype
    other than fp16, non-divisible shapes); the caller must decide on its own.
    """
    if dtype not in ("float16", "fp16"):
        return None
    if not (_divides(m, 64) and _divides(n, 64) and _divides(k, 64)):
        return None

    mn = m * n
    if _divides(m, 128) and _divides(n, 128):
        if mn >= _TILE_128_MIN_MN:
            return _CONFIG_128
        if mn >= _TILE_128_RELAXED_MN and k >= _TILE_128_RELAXED_K:
            return _CONFIG_128
    return _CONFIG_64
