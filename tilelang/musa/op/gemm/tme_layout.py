"""Configure hardware TME layout and swizzle granularity for SQMMA operands.

This is an AST transform, so user-written T.tma_copy + T.gemm kernels receive
the same hardware swizzle annotations as the automatic kernel builder.
Ambiguous partial-tile or multi-wait protocols are rejected instead of silently
consuming wrong layouts.
"""
from tilelang import language as T
from tvm import DataType, IRModule, tirx, arith


def prepare_tme_sqmma_layout(mod):
    result = IRModule(mod.functions, attrs=mod.attrs)
    for gv, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        calls = []
        tirx.stmt_functor.post_order_visit(func.body, lambda node: calls.append(node) if isinstance(node, tirx.Call) else None)
        layouts = {}
        analyzer = arith.Analyzer()
        for call in calls:
            if getattr(call.op, "name", "") not in ("tl.tileop.gemm", "tl.tileop.sqmma_gemm"):
                continue
            for region, k_major in ((call.args[0], not bool(call.args[3])), (call.args[1], bool(call.args[4]))):
                buffer = region.args[0].buffer
                if buffer.scope() not in ("shared", "shared.dyn"):
                    continue
                granularity = 16 if k_major else DataType(buffer.dtype).bits * 2
                previous = layouts.get(buffer.data)
                if previous is not None and previous != granularity:
                    raise ValueError("TME SQMMA operand is used with incompatible shared layouts")
                layouts[buffer.data] = granularity
        pending = {}
        for call in calls:
            if getattr(call.op, "name", "") != "tl.tileop.tma_copy":
                continue
            dst = call.args[1]
            buffer = dst.args[0].buffer
            if buffer.data not in layouts:
                continue
            barrier = call.annotations.get("barrier") if call.annotations else None
            if not isinstance(barrier, tirx.BufferLoad):
                raise ValueError("TME SQMMA load requires an explicit barrier")
            rank = len(buffer.shape)
            if rank not in (2, 3):
                raise ValueError("TME SQMMA supports a matrix or a single leading stage ring")
            minima = dst.args[0].indices
            extents = dst.args[2:]
            if any(not analyzer.can_prove_equal(i, 0) for i in minima[-2:]) or any(
                    not analyzer.can_prove_equal(e, s) for e, s in zip(extents[-2:], buffer.shape[-2:])):
                raise ValueError("TME SQMMA requires a complete contiguous operand tile")
            if rank == 3:
                if not analyzer.can_prove_equal(minima[0], barrier.indices[0]) or not analyzer.can_prove_equal(extents[0], 1):
                    raise ValueError("TME ring slot must equal its barrier index")
            elif not analyzer.can_prove_equal(barrier.buffer.shape[0], 1):
                raise ValueError("A non-ring TME operand requires one barrier")
            entries = pending.setdefault(barrier.buffer.data, {})
            entries[buffer.data] = (buffer, layouts[buffer.data])
        if not pending:
            continue
        waits = {key: 0 for key in pending}
        for call in calls:
            if getattr(call.op, "name", "") == "tl.mbarrier_wait_parity" and isinstance(call.args[0], tirx.BufferLoad):
                key = call.args[0].buffer.data
                if key in waits:
                    waits[key] += 1
        if any(count != 1 for count in waits.values()):
            raise ValueError("TME SQMMA requires one structured consumer wait per barrier")

        def rewrite(node):
            if not isinstance(node, tirx.Call):
                return None
            if getattr(node.op, "name", "") != "tl.tileop.tma_copy":
                return None
            dst = node.args[1]
            buffer = dst.args[0].buffer
            if buffer.data not in layouts:
                return None
            granularity = layouts[buffer.data]
            element_bytes = DataType(buffer.dtype).bits // 8
            tile_bytes = int(buffer.shape[-2]) * int(buffer.shape[-1]) * element_bytes
            if element_bytes not in (1, 2, 4) or granularity not in (16, 32, 64) or tile_bytes % 256:
                raise ValueError("TME SQMMA tile does not satisfy the 256-byte XOR layout contract")
            annotations = dict(node.annotations) if node.annotations else {}
            annotations["swizzle_granularity"] = tirx.IntImm("int32", granularity)
            return tirx.Call(node.dtype, node.op, node.args, annotations, node.span)

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.Call"])
        result[gv] = func.with_body(body)
    return result

