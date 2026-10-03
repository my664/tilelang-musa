"""Unit tests for the evidence-based MP31 SQMMA configuration planner."""

from tilelang.musa.op.gemm.planner import SqmmaConfig, recommend_sqmma_config


def test_large_square_uses_128():
    cfg = recommend_sqmma_config(2048, 2048, 2048)
    assert cfg == SqmmaConfig(128, 128, 64, 256, 1)


def test_small_square_uses_64():
    assert recommend_sqmma_config(1024, 1024, 1024) == SqmmaConfig(64, 64, 64, 128, 1)
    assert recommend_sqmma_config(512, 512, 4096) == SqmmaConfig(64, 64, 64, 128, 1)


def test_boundary_uses_k_when_output_is_marginal():
    # 1280^2 is 1.64e6 elements: 64x64 below K=640, 128x128 at/above.
    assert recommend_sqmma_config(1280, 1280, 512) == SqmmaConfig(64, 64, 64, 128, 1)
    assert recommend_sqmma_config(1280, 1280, 640) == SqmmaConfig(128, 128, 64, 256, 1)
    assert recommend_sqmma_config(1280, 1280, 1280) == SqmmaConfig(128, 128, 64, 256, 1)


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
