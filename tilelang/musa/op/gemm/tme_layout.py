"""Insert the SG_NONE-to-SQMMA layout conversion after DMA completion.

This is an AST transform, so user-written T.tma_copy + T.gemm kernels receive
the same conversion as the automatic kernel builder. Ambiguous partial-tile or
multi-wait protocols are rejected instead of silently consuming wrong layouts.
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
            if not isinstance(node, tirx.Evaluate) or not isinstance(node.value, tirx.Call):
                return None
            call = node.value
            if getattr(call.op, "name", "") != "tl.mbarrier_wait_parity" or not isinstance(call.args[0], tirx.BufferLoad):
                return None
            barrier = call.args[0]
            entries = pending.get(barrier.buffer.data)
            if entries is None:
                return None
            statements = [node]
            for buffer, granularity in entries.values():
                element_bytes = DataType(buffer.dtype).bits // 8
                tile_bytes = int(buffer.shape[-2]) * int(buffer.shape[-1]) * element_bytes
                if element_bytes not in (1, 2, 4) or granularity not in (16, 32, 64) or tile_bytes % 256:
                    raise ValueError("TME SQMMA tile does not satisfy the 256-byte XOR layout contract")
                indices = [barrier.indices[0], 0, 0] if len(buffer.shape) == 3 else [0, 0]
                base = tirx.BufferLoad(buffer, indices)
                pointer = T.access_ptr(base, "rw", extent=tile_bytes // element_bytes, ignore_last_ndim=2)
                name = f"tl::tme_sqmma_swizzle_inplace<{element_bytes}, {granularity}, {tile_bytes}>"
                statements.append(tirx.Evaluate(tirx.call_extern("handle", name, pointer)))
            return tirx.SeqStmt(statements)
        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.Evaluate"])
        result[gv] = func.with_body(body)
    return result
