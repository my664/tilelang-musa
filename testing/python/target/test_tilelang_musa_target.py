from tvm.target import Target
import pytest

from tilelang.musa.target import musa_arch_to_warp_size, normalize_musa_target


def test_musa_arch_to_warp_size():
    assert musa_arch_to_warp_size("mp_21") == 128
    assert musa_arch_to_warp_size("mp_22") == 128
    assert musa_arch_to_warp_size("mp_31") == 32
    assert musa_arch_to_warp_size("mp_41") is None
    assert musa_arch_to_warp_size("mp_11") is None


def test_normalize_musa_target_sets_arch_warp_size():
    for arch, expected_warp in (("mp_21", 128), ("mp_22", 128), ("mp_31", 32)):
        target = normalize_musa_target(Target({"kind": "musa", "arch": arch}))
        assert target is not None
        assert target.attrs["arch"] == arch
        assert int(target.attrs["thread_warp_size"]) == expected_warp


def test_tvm_target_canonicalizer_sets_explicit_arch_warp_size():
    for arch, expected_warp in (("mp_21", 128), ("mp_22", 128), ("mp_31", 32)):
        target = Target({"kind": "musa", "arch": arch})
        assert int(target.attrs["thread_warp_size"]) == expected_warp


def test_tvm_target_canonicalizer_rejects_unverified_arch():
    with pytest.raises(ValueError, match="no verified warp-size contract"):
        Target({"kind": "musa", "arch": "mp_41"})
