from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)


def _plan_copy_pipeline(mod: IRModule) -> IRModule:
    """Keep an explicitly synchronous copy policy during software pipelining.

    Without this, PipelinePlanning converts a sync candidate to cp.async and
    may fail vector injection or silently benchmark a different copy channel.
    Explicit TME kernels carry their own ring/barrier schedule.
    """
    planned = IRModule(mod.functions, attrs=mod.attrs)
    for gv, func in mod.functions.items():
        single = IRModule({gv: func})
        # Keep the measured default synchronous. A mechanism probe opts into
        # the SDK group protocol explicitly through the kernel attribute.
        use_async = bool(func.attrs and
                         func.attrs.get("mp31_pipeline_async", False))
        if func.attrs and func.attrs.get("mp31_copy_channel") == "sync" and not use_async:
            context = tilelang.transform.get_pass_context()
            config = dict(context.config)
            config["tirx.use_async_copy"] = False
            with tilelang.tvm.transform.PassContext(opt_level=context.opt_level,
                                                    config=config,
                                                    disabled_pass=list(context.disabled_pass)):
                single = tilelang.transform.PipelinePlanning()(single)
        else:
            single = tilelang.transform.PipelinePlanning()(single)
        planned[gv] = single[gv]
    return planned


def MUSAPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    from tilelang.musa.target import target_is_mp31
    from tilelang.musa.op.gemm.kernel import auto_tile_global_gemm

    if target_is_mp31(target):
        mod = auto_tile_global_gemm(mod)
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = _plan_copy_pipeline(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    if target_is_mp31(target):
        from tilelang.musa.op.gemm.tme_layout import prepare_tme_sqmma_layout
        mod = prepare_tme_sqmma_layout(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.musa.transform.LowerTMEIntrin()(mod)
    # Fold only provably point-to-point shared staging before vectorization,
    # StorageRewrite, and ThreadSync materialize the physical shared buffer.
    mod = tilelang.transform.EliminateRedundantSharedStaging()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.musa.transform.LowerLDGSTG()(mod)
    # Run immediately before host/device splitting. Public fast-div markers
    # remain opaque until here so a fast_divmod pair can share its quotient;
    # host-side magic parameter setup must not be inlined into device code.
    mod = tilelang.musa.transform.LowerFastDivmod()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(
        enable_aggressive_merge=enable_aggressive_merge,
        disable_reuse=disable_reuse,
    )(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


register_pipeline(PassPipeline("musa", MUSAPassPipelineBody))
