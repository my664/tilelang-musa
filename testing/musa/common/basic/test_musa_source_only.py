import os
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm


def _make_copy_kernel():
    @T.prim_func
    def kernel(
        src: T.Tensor((32, 32), T.float16),
        dst: T.Tensor((32, 32), T.float16),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((32, 32), T.float16)
            T.copy(src, shared)
            T.copy(shared, dst)

    return kernel


def _make_rmsnorm_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((2, 128), "bfloat16"),
        weight: T.Tensor((128,), "bfloat16"),
        y: T.Tensor((2, 128), "bfloat16"),
    ):
        with T.Kernel(2, threads=128) as row:
            shared = T.alloc_shared((1, 128), "bfloat16")
            x_local = T.alloc_fragment((1, 128), "bfloat16")
            xsq_f32 = T.alloc_fragment((1, 128), "float32")
            sumsq = T.alloc_fragment((1,), "float32")
            rrms = T.alloc_fragment((1,), "float32")
            T.copy(x[row, 0], shared)
            T.copy(shared, x_local)
            for _, col in T.Parallel(1, 128):
                value = T.cast(x_local[0, col], "float32")
                xsq_f32[0, col] = value * value
            T.reduce_sum(xsq_f32, sumsq, dim=1)
            rrms[0] = T.rsqrt(sumsq[0] / 128.0 + 1e-6)
            for _, col in T.Parallel(1, 128):
                x_local[0, col] = T.cast(
                    T.cast(x_local[0, col], "float32")
                    * rrms[0]
                    * T.cast(weight[col], "float32"),
                    "bfloat16",
                )
            T.copy(x_local, shared)
            T.copy(shared, y[row, 0])

    return kernel


def _make_softmax_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((2, 128), "float16"),
        y: T.Tensor((2, 128), "float16"),
    ):
        with T.Kernel(2, threads=128) as row:
            x_local = T.alloc_fragment((1, 128), "float16")
            exp_local = T.alloc_fragment((1, 128), "float32")
            max_value = T.alloc_fragment((1,), "float16")
            sum_value = T.alloc_fragment((1,), "float32")
            T.copy(x[row, 0], x_local)
            T.reduce_max(x_local, max_value, dim=1)
            for _, col in T.Parallel(1, 128):
                exp_local[0, col] = T.exp(
                    T.cast(x_local[0, col], "float32")
                    - T.cast(max_value[0], "float32")
                )
            T.reduce_sum(exp_local, sum_value, dim=1)
            for _, col in T.Parallel(1, 128):
                y[row, col] = T.cast(exp_local[0, col] / sum_value[0], "float16")

    return kernel


def _install_failing_mcc(fake_bin: Path, sentinel: Path) -> None:
    if os.name == "nt":
        script = fake_bin / "mcc.bat"
        script.write_text(f'@echo invoked>"{sentinel}"\r\n@exit /b 97\r\n', encoding="utf-8")
    else:
        script = fake_bin / "mcc"
        script.write_text(f'#!/bin/sh\nprintf invoked > "{sentinel}"\nexit 97\n', encoding="utf-8")
        script.chmod(0o755)


def test_musa_source_only_is_deterministic_and_does_not_invoke_mcc(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "mcc-was-invoked"
    _install_failing_mcc(fake_bin, sentinel)
    monkeypatch.setenv("PATH", os.pathsep.join((str(fake_bin), os.environ.get("PATH", ""))))
    monkeypatch.delenv("TVM_MUSA_COMPILER", raising=False)
    monkeypatch.delenv("TVM_MUSA_FLAGS", raising=False)

    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        first = tilelang.lower(
            _make_copy_kernel(),
            target=target,
            enable_device_compile=False,
        )
        second = tilelang.lower(
            _make_copy_kernel(),
            target=target,
            enable_device_compile=False,
        )

    assert "#include <musa.h>" in first.kernel_source
    assert "__global__" in first.kernel_source
    assert first.kernel_source == second.kernel_source
    assert not sentinel.exists(), "source-only lowering must not invoke mcc"

    with target, pytest.raises(RuntimeError, match=r"MTCompile failed \(code 97\)"):
        tilelang.lower(_make_copy_kernel(), target=target, enable_device_compile=True)
    assert sentinel.read_text(encoding="utf-8").strip() == "invoked"


def test_musa_source_only_reduction_emits_allreduce(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "mcc-was-invoked"
    _install_failing_mcc(fake_bin, sentinel)
    monkeypatch.setenv("PATH", os.pathsep.join((str(fake_bin), os.environ.get("PATH", ""))))
    monkeypatch.delenv("TVM_MUSA_COMPILER", raising=False)
    monkeypatch.delenv("TVM_MUSA_FLAGS", raising=False)

    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        lowered = tilelang.lower(
            _make_rmsnorm_kernel(),
            target=target,
            enable_device_compile=False,
        )

    assert "tl::WarpFirstAllReduce<" in lowered.kernel_source
    assert "__global__" in lowered.kernel_source
    assert not sentinel.exists(), "source-only reduction lowering must not invoke mcc"


def test_musa_source_only_softmax_emits_two_reductions(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "mcc-was-invoked"
    _install_failing_mcc(fake_bin, sentinel)
    monkeypatch.setenv("PATH", os.pathsep.join((str(fake_bin), os.environ.get("PATH", ""))))
    monkeypatch.delenv("TVM_MUSA_COMPILER", raising=False)
    monkeypatch.delenv("TVM_MUSA_FLAGS", raising=False)

    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        lowered = tilelang.lower(
            _make_softmax_kernel(),
            target=target,
            enable_device_compile=False,
        )

    assert lowered.kernel_source.count("tl::WarpFirstAllReduce<") >= 2
    assert "exp" in lowered.kernel_source.lower()
    assert not sentinel.exists(), "source-only softmax lowering must not invoke mcc"
