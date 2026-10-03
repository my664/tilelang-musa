"""Unit tests for the evidence-based MP31 SQMMA configuration planner."""

from tilelang.musa.op.gemm.planner import (SqmmaConfig, recommend_copy_channel,
                                           recommend_sqmma_config)


def test_large_square_uses_128():
    cfg = recommend_sqmma_config(2048, 2048, 2048)
    assert cfg == SqmmaConfig(128, 128, 64, 256, 1)


def test_small_square_uses_64():
    assert recommend_sqmma_config(1024, 1024, 1024) == SqmmaConfig(64, 64, 64, 128, 1)
    assert recommend_sqmma_config(512, 512, 4096) == SqmmaConfig(64, 64, 64, 128, 1)


def test_boundary_clauses_are_opt_in():
    # The core rule keeps 64x64 at M*N=1.64e6 (host MUPTI measured 64x64 faster).
    assert recommend_sqmma_config(1280, 1280, 640) == SqmmaConfig(64, 64, 64, 128, 1)
    # The container-derived clauses stay available explicitly.
    assert recommend_sqmma_config(1280, 1280, 640, boundary_clauses=True) == \
        SqmmaConfig(128, 128, 64, 256, 1)
    assert recommend_sqmma_config(1280, 1280, 1280, boundary_clauses=True) == \
        SqmmaConfig(128, 128, 64, 256, 1)


def test_long_k_clause_is_opt_in():
    # Container evidence: 1152^2 x 2048 measured 2.5% faster with 128x128.
    assert recommend_sqmma_config(1152, 1152, 2048) == SqmmaConfig(64, 64, 64, 128, 1)
    assert recommend_sqmma_config(1152, 1152, 2048, boundary_clauses=True) == \
        SqmmaConfig(128, 128, 64, 256, 1)
    # Smaller output with even longer K stays on 64x64 in both modes.
    assert recommend_sqmma_config(1024, 1024, 4096) == SqmmaConfig(64, 64, 64, 128, 1)
    assert recommend_sqmma_config(1024, 1024, 4096, boundary_clauses=True) == \
        SqmmaConfig(64, 64, 64, 128, 1)


def test_many_k_iterations_do_not_override_small_output():
    # 1024^2 with K=4096 was measured at 0.902 in favour of 64x64.
    assert recommend_sqmma_config(1024, 1024, 4096) == SqmmaConfig(64, 64, 64, 128, 1)


def test_rectangular_outputs():
    assert recommend_sqmma_config(1024, 2048, 1024) == SqmmaConfig(128, 128, 64, 256, 1)
    assert recommend_sqmma_config(2048, 4096, 2048) == SqmmaConfig(128, 128, 64, 256, 1)


def test_non_divisible_falls_back_to_64_or_none():
    # M=1088 is a multiple of 64 but not of 128: only 64x64 is legal.
    assert recommend_sqmma_config(1088, 1088, 1088) == SqmmaConfig(64, 64, 64, 128, 1)
    # 100 is not a multiple of 64: no measured-legal tile.
    assert recommend_sqmma_config(100, 128, 64) is None


def test_unsupported_dtype_returns_none():
    assert recommend_sqmma_config(2048, 2048, 2048, dtype="bfloat16") is None
    assert recommend_sqmma_config(2048, 2048, 2048, dtype="float32") is None


def test_copy_channel_rule():
    # MUPTI-measured break-even at 32 KB.
    assert recommend_copy_channel(512) == "direct"
    assert recommend_copy_channel(4 << 10) == "direct"
    assert recommend_copy_channel(32 << 10) == "tme"
    assert recommend_copy_channel(64 << 10) == "tme"
    assert recommend_copy_channel(160 << 10) == "tme"
