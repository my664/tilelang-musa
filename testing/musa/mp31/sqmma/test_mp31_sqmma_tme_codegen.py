"""CPU/source-only contracts. Run with the work repository's offline guard."""
import unittest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.musa.op.gemm.kernel import make_sqmma_gemm
from tilelang.musa.op.gemm.planner import SqmmaConfig


def lower(kernel):
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        return tilelang.lower(kernel, target=target, enable_device_compile=False)


def global_gemm(m, n, k, dtype="float16", clear=True, ta=False, tb=False):
    ashape = (k, m) if ta else (m, k)
    bshape = (n, k) if tb else (k, n)
    @T.prim_func
    def kernel(A: T.Tensor(ashape, dtype), B: T.Tensor(bshape, dtype),
               C: T.Tensor((m, n), "float32")):
        T.gemm(A, B, C, clear_accum=clear, transpose_A=ta, transpose_B=tb)
    return kernel


class TestMP31SourceOnly(unittest.TestCase):
    def test_user_written_tma_and_plain_gemm_get_layout_conversion(self):
        @T.prim_func
        def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"),
                   C: T.Tensor((64, 64), "float32")):
            with T.Kernel(1, threads=128):
                sa = T.alloc_shared((64, 64), "float16")
                sb = T.alloc_shared((64, 64), "float16")
                acc = T.alloc_fragment((64, 64), "float32")
                barrier = T.alloc_barrier(128)
                T.clear(acc)
                T.tma_copy(A, sa, barrier=barrier)
                T.tma_copy(B, sb, barrier=barrier)
                T.barrier_arrive(barrier)
                T.barrier_wait(barrier, 0)
                T.gemm(sa, sb, acc)
                T.copy(acc, C)
        src = lower(kernel).kernel_source
        self.assertIn("tl::tme_sqmma_swizzle_inplace<2, 16, 8192>", src)
        self.assertIn("tl::tme_sqmma_swizzle_inplace<2, 32, 8192>", src)
        self.assertLess(src.index("tl::tme_barrier_wait("), src.index("tl::tme_sqmma_swizzle_inplace<"))
        self.assertLess(src.index("tl::tme_sqmma_swizzle_inplace<"), src.index("tl::sqmma_ss<"))

    def test_sync_stages_do_not_change_copy_channel(self):
        for stages in (2, 3, 4):
            source = lower(make_sqmma_gemm(128, 256, 256,
                           config=SqmmaConfig(64, 64, 64, 256, stages))).kernel_source
            self.assertIn("tl::sqmma_ss<", source)
            self.assertNotIn("cp_async", source)
            self.assertNotIn("tme_load", source)

    def test_swizzle_is_bijective_involution(self):
        # Match the compiler's physical layout formula, including bf16/fp16
        # and K-major versus MN-major operand granularity. Every byte belongs
        # to exactly one disjoint swap pair, so there is no cross-thread race.
        for size in (4096, 8192, 16384):
            for granularity in (16, 32, 64):
                def partner(offset):
                    return (offset // 256 * 256 +
                            (((offset % 256) // granularity) ^ ((offset // 256) % (256 // granularity))) * granularity +
                            offset % granularity)
                mapped = [partner(i) for i in range(size)]
                self.assertEqual(sorted(mapped), list(range(size)))
                self.assertTrue(all(partner(partner(i)) == i for i in range(size)))
                self.assertTrue(all(partner(i) // 256 == i // 256 for i in range(size)))

    def test_untiled_plain_gemm_planner(self):
        for dims, tile in (((64, 128, 192), 64), ((1536, 1536, 256), 128)):
            with self.subTest(dims=dims):
                artifact = lower(global_gemm(*dims))
                self.assertIn(f"tl::sqmma_ss<{tile}, {tile}, 64", artifact.kernel_source)
                self.assertIn("tl_templates/musa/mp31/sqmma.h", artifact.kernel_source)

    def test_bf16_and_transpose(self):
        for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
            artifact = lower(global_gemm(128, 192, 256, "bfloat16", ta=ta, tb=tb))
            self.assertIn("tl::sqmma::bf16", artifact.kernel_source)

    def test_global_accumulation_is_preserved(self):
        artifact = lower(global_gemm(64, 64, 128, clear=False))
        src = artifact.kernel_source
        self.assertIn("tl::sqmma_ss<", src)
        # C is read into the fragment before GEMM rather than cleared.
        self.assertRegex(src, r"acc\[i\]\s*=\s*[^;]*C\)\[")

    def test_explicit_config(self):
        kernel = global_gemm(128, 128, 128).with_attr("mp31_tile_config", SqmmaConfig(64, 64, 64, 256, 1).as_dict())
        artifact = lower(kernel)
        func = next(iter(artifact.device_mod.functions.values()))
        self.assertEqual(int(func.attrs["thread_extent"]["threadIdx.x"]), 256)

    def test_illegal_shape_and_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiples of 64"):
            lower(global_gemm(65, 128, 128))
        with self.assertRaisesRegex(ValueError, "fp16/bf16"):
            lower(global_gemm(128, 128, 128, "float64"))
        with self.assertRaisesRegex(ValueError, "accumulator budget"):
            make_sqmma_gemm(256, 256, 128, config=SqmmaConfig(128, 128, 64, 128, 1))

    def test_rotating_tme_protocol(self):
        for stages in (1, 2, 3, 4):
            with self.subTest(stages=stages):
                kernel = make_sqmma_gemm(256, 256, 512, config=SqmmaConfig(64, 64, 64, 128, stages), channel="tme")
                artifact = lower(kernel)
                src = artifact.kernel_source
                for symbol in ("tme_load", "tme_barrier_add_trans", "tme_barrier_arrive", "tme_barrier_wait"):
                    self.assertIn("tl::" + symbol + "(", src)
                self.assertEqual(src.count("tl::tme_barrier_init_arrival("), stages)
                self.assertIn("tl_templates/musa/mp31/tme.h", src)
                self.assertIn("tl::tme_sqmma_swizzle_inplace<2, 16, 8192>", src)
                self.assertIn("tl::tme_sqmma_swizzle_inplace<2, 32, 8192>", src)
                # Consumer wait must precede tensor compute; a CTA fence must
                # precede reuse of its operand storage for the next transaction.
                wait = src.index("tl::tme_barrier_wait(")
                compute = src.index("tl::sqmma_ss<", wait)
                fence = src.index("__syncthreads()", compute)
                refill = src.index("tl::tme_load(", fence)
                self.assertLess(wait, compute)
                self.assertLess(fence, refill)
                # Check phase algebra at AST level, independent of whether
                # the final printer chooses %, bit-and, or a helper call.
                waits = []
                def visit(node):
                    if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.mbarrier_wait_parity":
                        waits.append(node)
                tvm.tirx.stmt_functor.post_order_visit(kernel.body, visit)
                self.assertEqual(len(waits), 1)
                phase = waits[0].args[1]
                variables = []
                tvm.tirx.stmt_functor.post_order_visit(phase, lambda v: variables.append(v) if isinstance(v, tvm.tirx.Var) else None)
                analyzer = tvm.arith.Analyzer()
                for iteration in range(8):
                    evaluated = tvm.tirx.stmt_functor.substitute(phase, {v: tvm.tirx.IntImm(v.dtype, iteration) for v in variables})
                    self.assertEqual(int(analyzer.simplify(evaluated)), (iteration // stages) % 2)

    def test_tme_short_k_tail(self):
        # More storage stages than K iterations must not issue out-of-range DMA.
        artifact = lower(make_sqmma_gemm(128, 128, 64, config=SqmmaConfig(64, 64, 64, 128, 4), channel="tme"))
        self.assertIn("tl::tme_load(", artifact.kernel_source)
        self.assertIn("tl::tme_barrier_wait(", artifact.kernel_source)


if __name__ == "__main__":
    unittest.main()
