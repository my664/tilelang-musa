"""Generic full-matrix MP31 GEMM construction with explicit copy completion.

Tiled callers retain control of their buffers. Untiled global GEMM callers use
the measured fp16 planner, or the same legal geometry for bf16 (unmeasured).
TME uses a bounded rotating ring: consume/wait, CTA reuse fence, then refill.
"""
from tilelang import language as T
from .planner import SqmmaConfig, recommend_sqmma_config


def make_sqmma_gemm(m, n, k, *, dtype="float16", out_dtype="float32",
                    config=None, channel="sync", clear_accum=True,
                    transpose_a=False, transpose_b=False, policy=None):
    policy = T.GemmWarpPolicy.Square if policy is None else policy
    if any(not isinstance(x, int) or x <= 0 or x % 64 for x in (m, n, k)):
        raise ValueError("MP31 automatic GEMM requires positive static M/N/K multiples of 64")
    if dtype not in ("float16", "bfloat16") or out_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("MP31 automatic GEMM requires fp16/bf16 inputs and floating output")
    if config is None:
        # bf16 shares the instruction geometry, but has no measured timing prior.
        config = recommend_sqmma_config(m, n, k, dtype="float16")
    if not isinstance(config, SqmmaConfig):
        raise ValueError("No legal MP31 GEMM configuration")
    bm, bn, bk, threads, stages = (config.block_m, config.block_n, config.block_k,
                                  config.threads, config.stages)
    if any(x <= 0 for x in (bm, bn, bk, threads, stages)):
        raise ValueError("Tile, threads and stages must be positive")
    if bm % 16 or bn % 16 or bk % 16 or threads % 128 or threads > 1024:
        raise ValueError("MP31 tile alignment or 128-thread squad alignment is invalid")
    if m % bm or n % bn or k % bk:
        raise ValueError("Explicit tiles must divide the problem dimensions")
    if (bm * bn + threads - 1) // threads > 64:
        raise ValueError("MP31 source-only accumulator budget exceeds 64 per thread")
    if stages * (bm + bn) * bk * 2 > 160 * 1024:
        raise ValueError("MP31 shared ring exceeds 160 KiB per block")
    if channel not in ("sync", "tme"):
        raise ValueError("Copy channel must be sync or tme")
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    a_tile = (bk, bm) if transpose_a else (bm, bk)
    b_tile = (bn, bk) if transpose_b else (bk, bn)
    iters = k // bk

    if channel == "sync":
        @T.prim_func
        def kernel(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype),
                   C: T.Tensor((m, n), out_dtype)):
            with T.Kernel(n // bn, m // bm, threads=threads) as (bx, by):
                sa = T.alloc_shared(a_tile, dtype)
                sb = T.alloc_shared(b_tile, dtype)
                acc = T.alloc_fragment((bm, bn), "float32")
                if clear_accum:
                    T.clear(acc)
                else:
                    T.copy(C[by * bm, bx * bn], acc)
                for ko in T.Pipelined(iters, num_stages=stages if stages > 1 else 0):
                    if transpose_a:
                        T.copy(A[ko * bk, by * bm], sa)
                    else:
                        T.copy(A[by * bm, ko * bk], sa)
                    if transpose_b:
                        T.copy(B[bx * bn, ko * bk], sb)
                    else:
                        T.copy(B[ko * bk, bx * bn], sb)
                    T.gemm(sa, sb, acc, transpose_A=transpose_a, transpose_B=transpose_b, policy=policy)
                T.copy(acc, C[by * bm, bx * bn])
    else:
        @T.prim_func
        def kernel(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype),
                   C: T.Tensor((m, n), out_dtype)):
            with T.Kernel(n // bn, m // bm, threads=threads) as (bx, by):
                sa = T.alloc_shared((stages, *a_tile), dtype)
                sb = T.alloc_shared((stages, *b_tile), dtype)
                barriers = T.alloc_barrier([threads] * stages)
                acc = T.alloc_fragment((bm, bn), "float32")
                if clear_accum:
                    T.clear(acc)
                else:
                    T.copy(C[by * bm, bx * bn], acc)
                for pre in T.serial(min(stages, iters)):
                    if transpose_a:
                        T.tma_copy(A[pre * bk:(pre + 1) * bk, by * bm:(by + 1) * bm], sa[pre, :, :], barrier=barriers[pre])
                    else:
                        T.tma_copy(A[by * bm:(by + 1) * bm, pre * bk:(pre + 1) * bk], sa[pre, :, :], barrier=barriers[pre])
                    if transpose_b:
                        T.tma_copy(B[bx * bn:(bx + 1) * bn, pre * bk:(pre + 1) * bk], sb[pre, :, :], barrier=barriers[pre])
                    else:
                        T.tma_copy(B[pre * bk:(pre + 1) * bk, bx * bn:(bx + 1) * bn], sb[pre, :, :], barrier=barriers[pre])
                    T.barrier_arrive(barriers[pre])
                for ko in T.serial(iters):
                    slot = ko % stages
                    T.barrier_wait(barriers[slot], (ko // stages) % 2)
                    T.gemm(sa[slot, :, :], sb[slot, :, :], acc,
                           transpose_A=transpose_a, transpose_B=transpose_b, policy=policy)
                    # Every squad must finish reading the slot before refilling.
                    T.sync_threads()
                    if ko + stages < iters:
                        if transpose_a:
                            T.tma_copy(A[(ko + stages) * bk:(ko + stages + 1) * bk, by * bm:(by + 1) * bm], sa[slot, :, :], barrier=barriers[slot])
                        else:
                            T.tma_copy(A[by * bm:(by + 1) * bm, (ko + stages) * bk:(ko + stages + 1) * bk], sa[slot, :, :], barrier=barriers[slot])
                        if transpose_b:
                            T.tma_copy(B[bx * bn:(bx + 1) * bn, (ko + stages) * bk:(ko + stages + 1) * bk], sb[slot, :, :], barrier=barriers[slot])
                        else:
                            T.tma_copy(B[(ko + stages) * bk:(ko + stages + 1) * bk, bx * bn:(bx + 1) * bn], sb[slot, :, :], barrier=barriers[slot])
                        T.barrier_arrive(barriers[slot])
                T.copy(acc, C[by * bm, bx * bn])
    return kernel.with_attr("mp31_auto_config", config.as_dict()).with_attr("mp31_copy_channel", channel)


def auto_tile_global_gemm(mod):
    """Expand a single, full global GEMM before launch materialization.

    Reject unsupported surrounding computation rather than silently dropping it.
    Explicit shared/fragment GEMMs pass through without changing their policy.
    """
    from tvm import tirx
    from tvm import IRModule
    mod = IRModule(mod.functions, attrs=mod.attrs)
    for gv, func in list(mod.functions.items()):
        if not isinstance(func, tirx.PrimFunc):
            continue
        globals_found = []
        def find(node):
            if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.gemm":
                globals_found.append(node)
        tirx.stmt_functor.post_order_visit(func.body, find)
        for call in globals_found:
            regions = call.args[:3]
            buffers = [r.args[0].buffer for r in regions]
            if any(b.scope() != "global" for b in buffers):
                continue
            body = func.body
            if isinstance(body, tirx.SBlockRealize):
                body = body.block.body
            if not isinstance(body, tirx.Evaluate) or not body.value.same_as(call):
                raise ValueError("MP31 automatic tiling requires a standalone full-matrix T.gemm")
            if len(func.params) != 3 or len(func.buffer_map) != 3:
                raise ValueError("MP31 automatic GEMM requires exactly three tensor parameters")
            for r, b in zip(regions, buffers):
                if len(b.shape) != 2 or (len(b.strides) and [int(s) for s in b.strides] != [int(b.shape[1]), 1]) or int(b.elem_offset) != 0:
                    raise ValueError("MP31 automatic GEMM requires contiguous rank-2 matrices")
                if any(int(i) != 0 for i in r.args[0].indices) or any(int(x) != int(y) for x, y in zip(r.args[2:], b.shape)):
                    raise ValueError("MP31 automatic tiling requires complete buffer regions")
            if buffers[0].dtype != buffers[1].dtype or buffers[2].dtype != "float32":
                raise ValueError("MP31 untiled GEMM requires matching fp16/bf16 inputs and fp32 C")
            if len({b.data for b in buffers}) != 3:
                raise ValueError("MP31 automatic GEMM does not support aliased operands")
            config_attr = func.attrs.get("mp31_tile_config") if func.attrs else None
            config = SqmmaConfig(**{str(key): int(value) for key, value in config_attr.items()}) if config_attr else None
            new = make_sqmma_gemm(int(call.args[5]), int(call.args[6]), int(call.args[7]),
                                  dtype=buffers[0].dtype, config=config,
                                  clear_accum=bool(call.args[9]),
                                  transpose_a=bool(call.args[3]), transpose_b=bool(call.args[4]), policy=call.args[8])
            # Keep the caller's ABI even if its parameters are not ordered A,B,C.
            order = [next(i for i, b in enumerate(buffers) if b.same_as(func.buffer_map[p])) for p in func.params]
            new = tirx.PrimFunc([new.params[i] for i in order], new.body,
                                new.ret_type, new.buffer_map, new.attrs, new.span)
            for key, value in func.attrs.items():
                new = new.with_attr(key, value)
            mod[gv] = new
    return mod
