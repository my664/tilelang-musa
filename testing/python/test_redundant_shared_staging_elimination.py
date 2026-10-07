"""Regression tests for the conservative shared-staging elimination pass.

The tests use source generation only.  Device correctness and Event timing are
kept in the S5000 experiment ledger, because a source-only test cannot prove
MUSA execution.
"""

import tilelang
import tilelang.language as T
from tilelang import tvm


def _source(kernel):
    with tvm.transform.PassContext(), tvm.target.Target({"kind": "musa", "arch": "mp_31"}):
        artifact = tilelang.lower(kernel, target={"kind": "musa", "arch": "mp_31"}, enable_device_compile=False)
    return artifact.kernel_source


def _rmsnorm(rows=2, cols=128, threads=128):
    @T.prim_func
    def kernel(
        x: T.Tensor((rows, cols), "bfloat16"),
        weight: T.Tensor((cols,), "bfloat16"),
        y: T.Tensor((rows, cols), "bfloat16"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            shared = T.alloc_shared((1, cols), "bfloat16")
            local = T.alloc_fragment((1, cols), "bfloat16")
            squares = T.alloc_fragment((1, cols), "float32")
            total = T.alloc_fragment((1,), "float32")
            scale = T.alloc_fragment((1,), "float32")
            T.copy(x[row, 0], shared)
            T.copy(shared, local)
            for _, col in T.Parallel(1, cols):
                value = T.cast(local[0, col], "float32")
                squares[0, col] = value * value
            T.reduce_sum(squares, total, dim=1)
            scale[0] = T.rsqrt(total[0] / float(cols) + 1e-6)
            for _, col in T.Parallel(1, cols):
                local[0, col] = T.cast(T.cast(local[0, col], "float32") * scale[0] * T.cast(weight[col], "float32"), "bfloat16")
            T.copy(local, shared)
            T.copy(shared, y[row, 0])

    return kernel


def _elementwise_add(rows=2, cols=128, threads=128):
    @T.prim_func
    def kernel(
        x: T.Tensor((rows, cols), "float16"),
        y: T.Tensor((rows, cols), "float16"),
        out: T.Tensor((rows, cols), "float16"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            shared = T.alloc_shared((1, cols), "float16")
            local = T.alloc_fragment((1, cols), "float16")
            T.copy(x[row, 0], shared)
            T.copy(shared, local)
            for _, col in T.Parallel(1, cols):
                local[0, col] = local[0, col] + y[row, col]
            T.copy(local, shared)
            T.copy(shared, out[row, 0])

    return kernel


def _layernorm(rows=2, cols=128, threads=128):
    @T.prim_func
    def kernel(
        x: T.Tensor((rows, cols), "float32"),
        out: T.Tensor((rows, cols), "float32"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            shared = T.alloc_shared((1, cols), "float32")
            local = T.alloc_fragment((1, cols), "float32")
            mean = T.alloc_fragment((1,), "float32")
            T.copy(x[row, 0], shared)
            T.copy(shared, local)
            T.reduce_sum(local, mean, dim=1)
            for _, col in T.Parallel(1, cols):
                local[0, col] = local[0, col] - mean[0] / float(cols)
            T.copy(local, shared)
            T.copy(shared, out[row, 0])

    return kernel


def test_rmsnorm_staging_is_removed_but_reduction_workspace_remains():
    source = _source(_rmsnorm())
    assert "void* shared =" not in source
    # The reduction itself may still require a shared workspace.  This test
    # therefore checks the named point-to-point staging, not all shared memory.
    assert "workspace" in source or "__syncthreads" in source


def test_elementwise_add_has_no_shared_staging():
    source = _source(_elementwise_add())
    assert "__shared__" not in source
    assert "__syncthreads" not in source


def test_layernorm_staging_is_removed():
    source = _source(_layernorm())
    assert "void* shared =" not in source
