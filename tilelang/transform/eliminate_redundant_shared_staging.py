"""Remove provably redundant global/shared/register copy chains.

The pass is deliberately conservative.  It only rewrites a shared buffer when
the same lexical loop nest performs both halves of a point-to-point copy and
the shared buffer has exactly the following access shape::

    global -> shared -> local/fragment
    local/fragment -> shared -> global

Indices must match after alpha-renaming the loop variables.  Any extra access,
different loop extent, non-direct value expression, or possible cross-thread
remapping leaves the buffer unchanged.  This keeps the pass an optimization,
never a semantic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm import ir, tirx
from tvm.tirx import (
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    Evaluate,
    For,
    PrimFunc,
    PyStmtExprMutator,
    SeqStmt,
    Stmt,
    StringImm,
    Var,
)
from tvm.tirx.stmt_functor import post_order_visit, substitute
from tvm.tirx.transform import prim_func_pass

from tilelang.utils.language import is_global, is_local, is_fragment, is_local_var, is_shared


def _same_expr(lhs, rhs) -> bool:
    try:
        return bool(ir.structural_equal(lhs, rhs))
    except Exception:
        return bool(tirx.analysis.expr_deep_equal(lhs, rhs))


def _is_register_buffer(buffer: Buffer) -> bool:
    return is_local(buffer) or is_fragment(buffer) or is_local_var(buffer)


def _is_shared_sync(stmt: Stmt) -> bool:
    if not isinstance(stmt, Evaluate) or not isinstance(stmt.value, Call):
        return False
    call = stmt.value
    if not call.op.same_as(tirx.builtin.tvm_storage_sync()):
        return False
    return bool(call.args) and isinstance(call.args[0], StringImm) and call.args[0].value in {"shared", "shared.dyn"}


@dataclass
class _WrappedStore:
    path: list[For]
    store: BufferStore


def _unwrap_store(stmt: Stmt) -> _WrappedStore | None:
    path: list[For] = []
    current = stmt
    while isinstance(current, For):
        path.append(current)
        current = current.body
    if isinstance(current, BufferStore):
        return _WrappedStore(path, current)
    return None


def _rebuild(path: list[For], store: BufferStore) -> Stmt:
    result: Stmt = store
    for loop in reversed(path):
        result = For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            result,
            loop.thread_binding,
            loop.annotations,
            loop.step,
        )
    return result


def _loop_mapping(producer: _WrappedStore, consumer: _WrappedStore) -> tuple[dict[Var, Var], dict[Var, Var]] | None:
    if len(producer.path) != len(consumer.path):
        return None
    producer_to_consumer: dict[Var, Var] = {}
    consumer_to_producer: dict[Var, Var] = {}
    for left, right in zip(producer.path, consumer.path):
        if left.kind != right.kind or not _same_expr(left.min, right.min) or not _same_expr(left.extent, right.extent):
            return None
        producer_to_consumer[left.loop_var] = right.loop_var
        consumer_to_producer[right.loop_var] = left.loop_var
    return producer_to_consumer, consumer_to_producer


def _indices_match(producer_indices, consumer_indices, consumer_to_producer: dict[Var, Var]) -> bool:
    if len(producer_indices) != len(consumer_indices):
        return False
    renamed = [substitute(index, consumer_to_producer) for index in consumer_indices]
    return all(_same_expr(left, right) for left, right in zip(producer_indices, renamed))


def _direct_load(store: BufferStore) -> BufferLoad | None:
    return store.value if isinstance(store.value, BufferLoad) else None


def _make_consumer_store(consumer: _WrappedStore, producer: _WrappedStore, source: Buffer) -> Stmt | None:
    mapping = _loop_mapping(producer, consumer)
    if mapping is None:
        return None
    producer_to_consumer, consumer_to_producer = mapping
    source_load = _direct_load(producer.store)
    consumer_load = _direct_load(consumer.store)
    if source_load is None or consumer_load is None or not consumer_load.buffer.same_as(producer.store.buffer):
        return None
    if not _indices_match(producer.store.indices, consumer_load.indices, consumer_to_producer):
        return None
    new_indices = [substitute(index, producer_to_consumer) for index in source_load.indices]
    new_value = BufferLoad(source, new_indices)
    new_store = BufferStore(consumer.store.buffer, new_value, consumer.store.indices)
    return _rebuild(consumer.path, new_store)


def _other_shared_access(stmt: Stmt, target: Buffer) -> bool:
    found = False

    def visit(node) -> None:
        nonlocal found
        if isinstance(node, (BufferLoad, BufferStore)) and is_shared(node.buffer) and not node.buffer.same_as(target):
            found = True

    post_order_visit(stmt, visit)
    return found


class _StagingMutator(PyStmtExprMutator):
    """Rewrite lexical sequences and remember buffers whose accesses vanished."""

    def __init__(self, shared_buffers: set[Buffer]):
        super().__init__()
        self.eliminated: set[Buffer] = set()
        self.shared_buffers = shared_buffers

    def visit_seq_stmt_(self, op: SeqStmt) -> Stmt:
        visited = [self.visit_stmt(stmt) for stmt in op.seq]
        changed = True
        while changed:
            changed = False
            for target in self._candidate_buffers(visited):
                rewritten = self._rewrite_buffer(visited, target)
                if rewritten is not None:
                    visited = rewritten
                    self.eliminated.add(target)
                    changed = True
                    break
        if not visited:
            return Evaluate(0)
        return visited[0] if len(visited) == 1 else SeqStmt(visited)

    @staticmethod
    def _candidate_buffers(stmts: list[Stmt]) -> list[Buffer]:
        result: list[Buffer] = []

        def visit(node) -> None:
            if isinstance(node, BufferStore) and is_shared(node.buffer) and not any(node.buffer.same_as(buf) for buf in result):
                result.append(node.buffer)
            elif isinstance(node, BufferLoad) and is_shared(node.buffer) and not any(node.buffer.same_as(buf) for buf in result):
                result.append(node.buffer)

        for stmt in stmts:
            post_order_visit(stmt, visit)
        return result

    def _rewrite_buffer(self, stmts: list[Stmt], target: Buffer) -> list[Stmt] | None:
        accesses: list[tuple[int, _WrappedStore]] = []
        for index, stmt in enumerate(stmts):
            wrapped = _unwrap_store(stmt)
            if wrapped is not None and (wrapped.store.buffer.same_as(target) or (isinstance(wrapped.store.value, BufferLoad) and wrapped.store.value.buffer.same_as(target))):
                accesses.append((index, wrapped))
        if len(accesses) != 4:
            return None
        input_pair = None
        output_pair = None
        for p_idx, producer in accesses:
            p_load = _direct_load(producer.store)
            if p_load is None:
                continue
            for c_idx, consumer in accesses:
                if c_idx <= p_idx or c_idx in {p_idx}:
                    continue
                c_load = _direct_load(consumer.store)
                if c_load is None:
                    continue
                if producer.store.buffer.same_as(target) and is_global(p_load.buffer) and is_register_buffer(consumer.store.buffer) and c_load.buffer.same_as(target):
                    candidate = _make_consumer_store(consumer, producer, p_load.buffer)
                    if candidate is not None:
                        input_pair = (p_idx, c_idx, candidate)
                if producer.store.buffer.same_as(target) and is_register_buffer(p_load.buffer) and is_global(consumer.store.buffer) and c_load.buffer.same_as(target):
                    candidate = _make_consumer_store(consumer, producer, p_load.buffer)
                    if candidate is not None:
                        output_pair = (p_idx, c_idx, candidate)
        if input_pair is None or output_pair is None:
            return None
        used = {input_pair[0], input_pair[1], output_pair[0], output_pair[1]}
        if len(used) != 4:
            return None
        result = list(stmts)
        for producer_index, consumer_index, replacement in sorted((input_pair, output_pair), reverse=True):
            if any(_other_shared_access(result[index], target) for index in range(producer_index + 1, consumer_index)):
                continue
            # Barriers in a pure staging gap have no remaining shared-memory
            # producer/consumer to order.  Barriers that also protect another
            # shared buffer are preserved.
            for index in range(consumer_index - 1, producer_index, -1):
                if _is_shared_sync(result[index]) and self.shared_buffers <= {target}:
                    result.pop(index)
                    if index < consumer_index:
                        consumer_index -= 1
            result[consumer_index] = replacement
            result.pop(producer_index)
        return result


class _CleanupMutator(PyStmtExprMutator):
    def __init__(self, removable: set[Buffer]):
        super().__init__()
        self.removable = removable

    def visit_alloc_buffer_(self, op):
        if any(op.buffer.same_as(buffer) for buffer in self.removable):
            return Evaluate(0)
        return op


def EliminateRedundantSharedStaging():
    """Create the fail-closed shared staging elimination pass."""

    def pass_fn(func: PrimFunc, mod, ctx):
        shared_buffers: set[Buffer] = set()

        def collect_shared(node) -> None:
            if isinstance(node, (BufferLoad, BufferStore)) and is_shared(node.buffer):
                shared_buffers.add(node.buffer)

        post_order_visit(func.body, collect_shared)
        mutator = _StagingMutator(shared_buffers)
        body = mutator.visit_stmt(func.body)
        # Only remove declarations after a whole-function access check.  A
        # partially matched buffer remains allocated, preserving semantics.
        remaining: set[Buffer] = set()

        def visit(node) -> None:
            if isinstance(node, (BufferLoad, BufferStore)):
                for buffer in mutator.eliminated:
                    if node.buffer.same_as(buffer):
                        remaining.add(buffer)

        post_order_visit(body, visit)
        removable = mutator.eliminated - remaining
        if removable:
            body = _CleanupMutator(removable).visit_stmt(body)
        return func.with_body(body)

    return prim_func_pass(pass_fn, opt_level=0)
