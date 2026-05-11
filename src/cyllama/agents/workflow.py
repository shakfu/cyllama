"""DAG-based agent workflow orchestration.

Phases 1 + 2 + 3 + 4 + 5 landed: Layer B core, Layer C decorator sugar,
streaming + helpers + visualization, contracts + reducers +
typed-state generic, and ``AgentProtocol`` compliance + sub-workflow
event nesting.

This module provides the canonical runtime model for cyllama workflows:
an explicit StateGraph with typed state, static and conditional edges,
parallel execution of independent nodes, and integration with the
existing agent primitives (`AgentProtocol`, `Tool`, `AgentEvent`).

**Layer B** (canonical runtime) -- explicit ``add_node`` /
``add_edge`` / ``add_conditional_edge`` / ``set_entry`` /
``set_exit`` calls building a :class:`Workflow`, then :meth:`compile`
and :meth:`run`. State is a dict (or typed via a ``TypedDict``);
nodes return partial state updates.

**Layer C** (decorator sugar) -- ``@flow.node`` / ``@flow.route``
decorators with parameter-name dependency inference. Desugars
internally to the same Layer-B calls; the two layers are
interoperable on the same :class:`Workflow` object.

See ``docs/agents/workflow.md`` for the full implementation
specification.

Public surface (Phases 1-3):

- :class:`Workflow` -- main entry point. Both ``add_node(name, fn)``
  Layer-B form and ``@flow.node`` / ``flow.add_node(fn)`` Layer-C
  form. ``@flow.route(after=...)`` decorator for conditional edges.
  ``run`` / ``arun`` / ``stream`` / ``astream`` execution surfaces.
  ``to_mermaid`` / ``to_dot`` / ``dry_run`` for inspection.
- :class:`CompiledWorkflow` -- validated, runnable graph.
- :class:`WorkflowResult` -- run output (state + success + error +
  events).
- :class:`DryRunPlan` -- static execution plan from ``dry_run``.
- :class:`WorkflowDefinitionError`,
  :class:`WorkflowExecutionError`,
  :class:`WorkflowRoutingError` -- error categories.
- :data:`END` -- sentinel router return value, terminates a branch.
- :func:`agent_node` -- wrap an ``AgentProtocol`` (including another
  :class:`Workflow` / :class:`CompiledWorkflow`) as a workflow node.
  When the inner agent is a workflow, inner events are forwarded into
  the outer stream with ``source`` and ``parent_event_id`` set.
- :func:`tool_node` -- wrap a ``Tool`` as a workflow node.
- :func:`workflow_node` -- wrap another :class:`CompiledWorkflow` as a
  node with event forwarding.

Both :class:`Workflow` and :class:`CompiledWorkflow` satisfy the
:class:`AgentProtocol` structural contract (``run`` / ``stream`` /
``arun`` accepting a positional ``task: str`` plus a ``metrics``
property), so they plug into :func:`agent_as_tool`,
:class:`TieredAgentTeam`, and :class:`ReflectionLoop` unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from .contract import ContractPolicy, ContractViolation
from .types import AgentEvent, AgentMetrics, AgentResult, EventType


__all__ = [
    "Workflow",
    "CompiledWorkflow",
    "WorkflowResult",
    "DryRunPlan",
    "WorkflowDefinitionError",
    "WorkflowExecutionError",
    "WorkflowRoutingError",
    "END",
    "agent_node",
    "tool_node",
    # Phase 4 surface
    "WorkflowInvariant",
    "WorkflowExecutionState",
    "reducer",
    # Phase 5 surface
    "workflow_node",
]


StateT = TypeVar("StateT", bound=Dict[str, Any])


# ---------------------------------------------------------------------------
# Sentinel for "terminate this branch"
# ---------------------------------------------------------------------------


class _EndSentinel:
    """Singleton sentinel returned from conditional routers to terminate
    a branch without dispatching to a downstream node. Compared by
    identity; never instantiated by users."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "END"

    def __bool__(self) -> bool:
        # Avoid accidental truthiness tests treating END as truthy.
        return False


END: _EndSentinel = _EndSentinel()


# ---------------------------------------------------------------------------
# Reducers (Phase 4)
#
# A reducer is any callable ``(existing, update) -> merged`` used to combine
# state values when multiple nodes write the same key. Without a reducer
# registered for a key, two nodes writing it is a structural error caught
# at runtime (Phase 4 deferred from compile-time per the design doc since
# Layer-B node bodies are opaque to static analysis).
# ---------------------------------------------------------------------------


def _reducer_append(existing: Any, update: Any) -> Any:
    """Reducer: append a single value to a list.

    ``existing`` may be ``None`` (treated as the empty list) or a list;
    ``update`` is the new element. Returns a new list -- the existing
    one is not mutated.
    """
    if existing is None:
        return [update]
    return list(existing) + [update]


def _reducer_extend(existing: Any, update: Any) -> Any:
    """Reducer: extend a list with another list.

    Both inputs may be ``None`` (treated as the empty list). Returns
    a new list.
    """
    base = list(existing) if existing is not None else []
    tail = list(update) if update is not None else []
    return base + tail


def _reducer_merge_dict(existing: Any, update: Any) -> Any:
    """Reducer: ``dict.update``-style merge.

    Both inputs may be ``None`` (treated as the empty dict). ``update``
    keys win over ``existing`` for collisions. Returns a new dict.
    """
    base = dict(existing) if existing is not None else {}
    tail = dict(update) if update is not None else {}
    base.update(tail)
    return base


def _reducer_add(existing: Any, update: Any) -> Any:
    """Reducer: numeric accumulation. ``existing`` defaults to 0 if ``None``."""
    if existing is None:
        return update
    return existing + update


def _reducer_last(existing: Any, update: Any) -> Any:
    """Reducer: last-writer-wins. ``existing`` is discarded.

    Explicit opt-in to the previous (Phase 1-3) default behavior --
    register this on a key to allow multiple nodes to write it
    without triggering the multi-writer detection.
    """
    return update


class _ReducerNamespace:
    """Namespace exposing the built-in reducers as attributes.

    Usage::

        from cyllama.agents.workflow import Workflow, reducer

        flow = Workflow(State, reducers={
            "messages": reducer.append,
            "tags": reducer.extend,
            "metadata": reducer.merge_dict,
            "score": reducer.add,
        })

    Custom reducers are plain callables matching
    ``(existing, update) -> merged``; no inheritance or registration
    is required -- just pass the callable directly.
    """

    append = staticmethod(_reducer_append)
    extend = staticmethod(_reducer_extend)
    merge_dict = staticmethod(_reducer_merge_dict)
    add = staticmethod(_reducer_add)
    last = staticmethod(_reducer_last)


reducer = _ReducerNamespace()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WorkflowDefinitionError(ValueError):
    """Raised by :meth:`Workflow.compile` when the workflow definition
    is structurally invalid (unknown node refs, cycle in static edges,
    multiple writers without a reducer, missing entry, etc.).

    All compile-time failures use this exception. Runtime failures
    raise :class:`WorkflowExecutionError` or
    :class:`WorkflowRoutingError`.
    """


class WorkflowExecutionError(RuntimeError):
    """Raised at runtime for structural failures inside a workflow run:
    ``max_steps`` exceeded, an unresolvable dependency, etc.

    User node bodies that raise produce a captured-error
    :class:`WorkflowResult` rather than propagating; this exception is
    reserved for the framework's own runtime invariants.
    """


class WorkflowRoutingError(RuntimeError):
    """Raised when a conditional router returns a value not in the
    declared ``edge_map``, or when ``edge_map`` is None and the
    returned name is not a registered node (and not :data:`END`).
    """


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    """A compiled node: name, callable, optional per-node timeout.

    The callable's Layer-B signature is ``(state: dict) -> dict``,
    returning a *partial* state update to merge.
    """

    name: str
    fn: Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]
    timeout: Optional[float] = None


@dataclass(frozen=True)
class _ConditionalEdge:
    """A conditional outgoing edge from one node.

    The router is called with current state after ``from_node``
    completes; its return value is either a target node name (looked
    up in ``edge_map`` if provided, used directly otherwise) or
    :data:`END` to terminate the branch.

    A node may have at most one conditional outgoing edge. Multiple
    targets are encoded via ``edge_map``, not via multiple
    ``_ConditionalEdge`` objects.
    """

    from_node: str
    router: Callable[[Dict[str, Any]], Any]
    edge_map: Optional[Dict[Any, Union[str, _EndSentinel]]]


@dataclass(frozen=True)
class _LayerCMeta:
    """Layer-C metadata for a node registered via the decorator API.

    Captured at decoration time so that compile-time edge inference and
    runtime input-requirement validation can run independently of the
    wrapped function.
    """

    # Original (undecorated) function name, kept for diagnostics.
    fn_name: str
    # Ordered list of the original function's parameter names. Each
    # matches a state key at runtime (either the return of another
    # node, or a workflow input from ``run(**initial_state)``).
    param_names: Tuple[str, ...]
    # Return annotation as a string ("list[str]" etc.) or None when
    # unannotated. Used by :attr:`Workflow.derived_state_schema`.
    return_annotation: Any


@dataclass(frozen=True)
class _LayerCRouterMeta:
    """Layer-C metadata for a router registered via ``@flow.route``."""

    fn_name: str
    param_names: Tuple[str, ...]


def _extract_param_names(fn: Callable[..., Any], *, role: str) -> List[str]:
    """Return ordered parameter names of ``fn``.

    Rejects functions whose signature includes ``*args``, ``**kwargs``,
    positional-only parameters with no name, or `self` (suggesting an
    unbound method). Keyword-only and positional-or-keyword params are
    accepted; defaults are allowed.

    ``role`` is used purely for diagnostic messages
    ("Layer-C {role} {fn_name!r} ...").
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise WorkflowDefinitionError(
            f"Layer-C {role} {getattr(fn, '__name__', '<anon>')!r}: could not introspect signature ({e})"
        ) from e

    names: List[str] = []
    for p_name, p in sig.parameters.items():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise WorkflowDefinitionError(
                f"Layer-C {role} {getattr(fn, '__name__', '<anon>')!r}: "
                f"signature has *{p_name} or **{p_name}; Layer C requires "
                f"explicit named parameters"
            )
        names.append(p_name)
    return names


def _extract_return_annotation(fn: Callable[..., Any]) -> Any:
    """Return the function's resolved return annotation, or None if absent.

    Used by :attr:`Workflow.derived_state_schema`. Resolves forward
    references (e.g. annotations evaluated lazily under
    ``from __future__ import annotations``) via
    :func:`typing.get_type_hints`. Falls back to the raw annotation
    when resolution fails (forward ref to a name not in scope, etc.).
    """
    import typing as _t

    try:
        hints = _t.get_type_hints(fn)
        if "return" in hints:
            return hints["return"]
    except Exception:
        # get_type_hints can fail on unresolvable forward refs; fall
        # back to the raw signature annotation.
        pass

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return None
    return ret


def _layer_c_node_wrapper(
    user_fn: Callable[..., Any],
    node_name: str,
    param_names: List[str],
) -> Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]:
    """Build the Layer-B wrapper for a Layer-C node.

    The wrapper:
    1. Pulls each parameter from state by name.
    2. Calls ``user_fn(**kwargs)``.
    3. Returns ``{node_name: result}`` so the value lands in state under
       the node's own name.

    Sync functions get a sync wrapper; async functions get an async
    wrapper. The CompiledWorkflow runtime dispatches each appropriately.
    """
    is_async = asyncio.iscoroutinefunction(user_fn)

    if is_async:

        async def async_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            kwargs = {p: state[p] for p in param_names}
            result = await user_fn(**kwargs)
            return {node_name: result}

        return async_wrapper

    def sync_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {p: state[p] for p in param_names}
        result = user_fn(**kwargs)
        return {node_name: result}

    return sync_wrapper


def _layer_c_router_wrapper(
    user_fn: Callable[..., Any],
    param_names: List[str],
) -> Callable[[Dict[str, Any]], Any]:
    """Build the conditional-edge wrapper for a Layer-C router.

    Routers don't return state updates -- they return the next node
    name (or :data:`END`). The wrapper just binds params from state
    and forwards the user function's return value verbatim.
    """

    def wrapper(state: Dict[str, Any]) -> Any:
        kwargs = {p: state[p] for p in param_names}
        return user_fn(**kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult(Generic[StateT]):
    """Result of a workflow run.

    Attributes:
        state: Final merged state. Same shape as the workflow's state
            schema (or a plain dict if no schema was supplied).
        success: True if the workflow ran to completion without a
            captured error. False if a node raised, a router was
            invalid, or ``max_steps`` was hit.
        error: Human-readable error message when ``success=False``.
        metrics: Per-run timing and counters (total time, node count,
            error count). See :class:`AgentMetrics`.
        nodes_run: Ordered list of node names that executed (Phase 1
            convenience; same info is recoverable from the metrics).
    """

    state: StateT
    success: bool
    error: Optional[str] = None
    metrics: Optional[AgentMetrics] = None
    nodes_run: List[str] = field(default_factory=list)
    # Full event log: WORKFLOW_START, per-node NODE_START/NODE_END,
    # WORKFLOW_END, plus any ERROR events captured along the way.
    # Populated by ``run()`` from the same internal event source that
    # ``stream()`` yields, so the two views are consistent.
    events: List[AgentEvent] = field(default_factory=list)
    # Phase 5: state key whose final value is treated as the "answer"
    # for AgentResult-shape consumers (``agent_as_tool``,
    # ``ReflectionLoop``). Populated by ``Workflow`` / ``CompiledWorkflow``
    # at result-construction time from their ``task_param`` and exit set;
    # defaults to ``None`` for direct construction.
    answer_key: Optional[str] = None

    @property
    def answer(self) -> str:
        """AgentResult-shape adapter: final state value at ``answer_key``.

        Resolution order: explicit ``answer_key`` field, then the sole
        exit-node entry in state, then the empty string. Always returns
        a ``str`` (stringifying non-string values) so consumers like
        :func:`agent_as_tool` can rely on the contract.
        """
        if self.answer_key is not None and self.answer_key in self.state:
            return str(self.state[self.answer_key])
        # Fall back to the last node's state entry if it's stringifiable.
        if self.nodes_run:
            last = self.nodes_run[-1]
            if last in self.state:
                return str(self.state[last])
        return ""

    @property
    def steps(self) -> List[AgentEvent]:
        """AgentResult-shape adapter: alias for :attr:`events`."""
        return self.events

    @property
    def iterations(self) -> int:
        """AgentResult-shape adapter: number of nodes that ran."""
        return len(self.nodes_run)


@dataclass(frozen=True)
class DryRunPlan:
    """Static execution plan returned by :meth:`Workflow.dry_run`.

    Shows the order in which nodes would execute (broken into
    topological levels over static edges) and which nodes are
    reachable only through conditional routing (so their inclusion
    in any given run depends on runtime data). Useful for validating
    a workflow's shape without paying the cost of actually running it.

    Attributes:
        levels: List of node-name groups in execution order.
            Within a group, nodes run concurrently in real execution.
        conditional_nodes: Set of nodes that have a conditional
            incoming edge -- they may not run on every workflow run.
        entry: Entry node name.
        exits: Frozenset of exit node names.
        inputs_required: Sorted list of state keys that must be
            supplied via ``run(**initial_state)`` (Layer-C input
            requirements).
    """

    levels: Tuple[Tuple[str, ...], ...]
    conditional_nodes: frozenset[str]
    entry: str
    exits: frozenset[str]
    inputs_required: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Phase 4: WorkflowInvariant + execution state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowExecutionState:
    """View of a running workflow's state, passed to invariant predicates.

    Augments the user-facing state dict with framework-tracked counters
    so invariants can express budgets and rate limits without the user
    threading those values through state manually.

    Attributes:
        state: The current workflow state (read-only snapshot; mutations
            don't propagate back to the runtime).
        elapsed_ms: Wall-clock since the workflow started.
        node_count: Nodes completed so far.
        error_count: Errors captured so far (always 0 until terminal
            failure since the workflow currently aborts on first error;
            preserved for future per-node retry semantics).
        estimated_cost_usd: User-tracked cost accumulator. Defaults to
            0.0; the framework doesn't know about costs, but user-side
            agent integrations can update ``state[".cost_usd"]`` (or
            similar) and have an invariant check it. The dedicated
            field exists so the invariant API doesn't depend on a
            state-key convention.
        nodes_run: Ordered list of completed node names.
    """

    state: Dict[str, Any]
    elapsed_ms: float
    node_count: int
    error_count: int
    estimated_cost_usd: float
    nodes_run: Tuple[str, ...]


@dataclass(frozen=True)
class WorkflowInvariant:
    """A workflow-level invariant checked after each node completes.

    The predicate receives a :class:`WorkflowExecutionState` view and
    returns True if the invariant holds. On False, the workflow emits
    a ``CONTRACT_VIOLATION`` event; under
    :class:`~cyllama.agents.ContractPolicy.ENFORCE` the workflow also
    terminates.

    Attributes:
        predicate: ``Callable[[WorkflowExecutionState], bool]``.
        message: Human-readable description of what the invariant
            checks. Surfaced in the violation event's metadata and
            (under ENFORCE) in the captured error message.
        policy: Optional per-invariant policy override. ``None`` means
            "inherit from ``Workflow.policy``".

    Example::

        from cyllama.agents.workflow import Workflow, WorkflowInvariant
        from cyllama.agents import ContractPolicy

        flow = Workflow(
            invariants=[
                WorkflowInvariant(
                    predicate=lambda s: s.elapsed_ms < 30_000,
                    message="total time < 30s",
                ),
                WorkflowInvariant(
                    predicate=lambda s: s.node_count < 50,
                    message="no more than 50 nodes",
                ),
            ],
            policy=ContractPolicy.ENFORCE,
        )
    """

    predicate: Callable[["WorkflowExecutionState"], bool]
    message: str
    policy: Optional[ContractPolicy] = None


# ---------------------------------------------------------------------------
# Workflow (definition-time)
# ---------------------------------------------------------------------------


class Workflow(Generic[StateT]):
    """A DAG-based agent workflow.

    Phase 1 exposes the Layer-B (explicit StateGraph) API only. Phase 2
    will add the Layer-C decorator API (``@flow.node`` /
    ``@flow.route``) on top.

    Args:
        state_schema: Optional ``TypedDict`` (or any subscriptable class)
            describing the workflow state. When omitted, state is a
            plain ``dict[str, Any]`` with no validation. Phase 1 uses
            the schema only for the ``state_schema`` attribute exposed
            on the compiled workflow; deeper schema-driven validation
            lands in Phase 4 with the reducer machinery.
        max_steps: Hard cap on iterations of the workflow scheduler.
            Each iteration runs one or more ready nodes in parallel
            and then re-evaluates the ready set. The cap exists to
            bound conditional-edge cycles. Default: 100.

    Use :meth:`add_node` / :meth:`add_edge` /
    :meth:`add_conditional_edge` / :meth:`set_entry` / :meth:`set_exit`
    to build the graph. Call :meth:`compile` to validate, or call
    :meth:`run` / :meth:`arun` to compile implicitly and execute.
    """

    def __init__(
        self,
        state_schema: Optional[type] = None,
        *,
        max_steps: int = 100,
        reducers: Optional[Dict[str, Callable[[Any, Any], Any]]] = None,
        invariants: Optional[List[WorkflowInvariant]] = None,
        policy: ContractPolicy = ContractPolicy.ENFORCE,
        task_param: str = "task",
        answer_key: Optional[str] = None,
    ) -> None:
        self._state_schema = state_schema
        self._max_steps = max_steps
        # Phase 4 -- reducer registry and invariant set.
        self._reducers: Dict[str, Callable[[Any, Any], Any]] = dict(reducers or {})
        self._invariants: List[WorkflowInvariant] = list(invariants or [])
        self._policy = policy
        # Phase 5 -- AgentProtocol compliance: positional ``run(task)``
        # binds ``task`` into ``state[task_param]``; the final value of
        # ``state[answer_key]`` (or, if unset, the sole exit node) is
        # what ``WorkflowResult.answer`` returns. ``answer_key=None``
        # delays the choice until ``compile()`` -- if exactly one exit
        # is registered, that exit's name is used.
        self._task_param = task_param
        self._answer_key = answer_key
        # Cache of the most-recent run's AgentMetrics, exposed via
        # ``metrics`` property to satisfy AgentProtocol.
        self._last_metrics: Optional[AgentMetrics] = None

        # name -> Node
        self._nodes: Dict[str, _Node] = {}
        # from_node -> [to_node, ...] (static edges)
        self._static_edges: Dict[str, List[str]] = {}
        # to_node -> [from_node, ...] (reverse index over static edges)
        self._static_reverse: Dict[str, List[str]] = {}
        # from_node -> ConditionalEdge (at most one per from_node)
        self._conditional_edges: Dict[str, _ConditionalEdge] = {}
        self._entry: Optional[str] = None
        self._exits: Set[str] = set()

        # Layer-C tracking: nodes registered via @flow.node / add_node(fn)
        # keyed by node name. Used at compile time to infer static edges
        # from parameter-name matches; at run time to validate required
        # initial_state inputs.
        self._layer_c_meta: Dict[str, _LayerCMeta] = {}
        # Layer-C routers registered via @flow.route, keyed by from_node.
        self._layer_c_router_meta: Dict[str, _LayerCRouterMeta] = {}

        # Cached compile output; invalidated by any mutation.
        self._compiled: Optional["CompiledWorkflow[StateT]"] = None

    # -- Graph building ----------------------------------------------------

    def add_node(
        self,
        name_or_fn: Union[str, Callable[..., Any]],
        fn: Optional[Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]] = None,
        *,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Register a node. Accepts both Layer-B and Layer-C forms.

        **Layer B** (canonical): ``add_node(name, fn)``, where ``fn`` is
        a callable with signature ``(state: dict) -> dict | None``
        returning a partial state update.

        **Layer C** (sugar): ``add_node(fn, name=None)``, where ``fn``
        has typed named parameters. The framework introspects ``fn``'s
        signature; parameter names bind from state at run time, and the
        return value is stored in state under ``name`` (or ``fn.__name__``
        if ``name`` is omitted). The same node is accessible via the
        ``@flow.node`` decorator -- this form is for programmatic
        registration of pre-defined functions.

        Args:
            name_or_fn: Either the node name (Layer B) or the function
                to register (Layer C).
            fn: Layer-B callable ``(state) -> dict``. Required if the
                first argument is a string; ignored otherwise.
            name: Layer-C node name override; ignored if the first
                argument is a string.
            timeout: Optional per-node timeout in seconds. Raises
                :class:`asyncio.TimeoutError` on exceedance; the
                workflow converts that to a captured error.

        Raises:
            WorkflowDefinitionError: If ``name`` is already registered,
                is empty, or the function signature includes ``*args``
                or ``**kwargs`` (Layer-C nodes require explicit named
                parameters).
            TypeError: If ``name_or_fn`` is neither a string nor a
                callable.
        """
        if isinstance(name_or_fn, str):
            # Layer-B form: add_node(name, fn).
            node_name = name_or_fn
            node_fn = fn
            if node_fn is None:
                raise TypeError(f"add_node({node_name!r}, ...): fn argument is required for the Layer-B form")
            self._add_layer_b_node(node_name, node_fn, timeout=timeout)
        elif callable(name_or_fn):
            # Layer-C form: add_node(fn) or add_node(fn, name="x").
            user_fn = name_or_fn
            derived_name: Optional[str] = name or getattr(user_fn, "__name__", None)
            if not derived_name:
                raise WorkflowDefinitionError("Layer-C node has no derivable name; pass name= explicitly")
            self._add_layer_c_node(derived_name, user_fn, timeout=timeout)
        else:
            raise TypeError(
                f"add_node: first argument must be a node name (str) or a callable; got {type(name_or_fn).__name__}"
            )

    def _add_layer_b_node(
        self,
        name: str,
        fn: Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]],
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Internal: register a Layer-B node (state -> partial dict)."""
        if not name:
            raise WorkflowDefinitionError("node name must be non-empty")
        if name in self._nodes:
            raise WorkflowDefinitionError(f"duplicate node name: {name!r}")
        self._nodes[name] = _Node(name=name, fn=fn, timeout=timeout)
        self._compiled = None

    def _add_layer_c_node(
        self,
        name: str,
        user_fn: Callable[..., Any],
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Internal: register a Layer-C node (kwargs bound from state).

        Introspects the user function's signature, builds a Layer-B
        wrapper that pulls each parameter from state by name and stores
        the return value under ``name``, and records metadata for
        compile-time edge inference.
        """
        if not name:
            raise WorkflowDefinitionError("node name must be non-empty")
        if name in self._nodes:
            raise WorkflowDefinitionError(f"duplicate node name: {name!r}")

        param_names = _extract_param_names(user_fn, role="node")
        return_annotation = _extract_return_annotation(user_fn)
        wrapper = _layer_c_node_wrapper(user_fn, name, param_names)

        self._nodes[name] = _Node(name=name, fn=wrapper, timeout=timeout)
        self._layer_c_meta[name] = _LayerCMeta(
            fn_name=getattr(user_fn, "__name__", name),
            param_names=tuple(param_names),
            return_annotation=return_annotation,
        )
        self._compiled = None

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Declare a static edge: ``to_node`` runs after ``from_node``.

        Both endpoints are validated at :meth:`compile` time, not
        immediately, so the order of ``add_node`` / ``add_edge`` calls
        doesn't matter.

        A single ``from_node`` may have multiple outgoing static
        edges (fan-out) and a single ``to_node`` may have multiple
        incoming static edges (fan-in -- ``to_node`` waits for all).
        """
        self._static_edges.setdefault(from_node, []).append(to_node)
        self._static_reverse.setdefault(to_node, []).append(from_node)
        self._compiled = None

    def add_conditional_edge(
        self,
        from_node: str,
        router: Callable[[Dict[str, Any]], Any],
        edge_map: Optional[Dict[Any, Union[str, _EndSentinel]]] = None,
    ) -> None:
        """Declare a conditional outgoing edge.

        After ``from_node`` completes, ``router(state)`` is called.
        Its return value is mapped through ``edge_map`` (if supplied)
        to the next node name, or used directly as a node name (if
        ``edge_map`` is None). The sentinel :data:`END` may be returned
        to terminate this branch without dispatching to a downstream
        node.

        A node may have at most one conditional outgoing edge.
        Calling this twice for the same ``from_node`` raises
        :class:`WorkflowDefinitionError`.

        A node may have *either* static outgoing edges or a conditional
        outgoing edge, not both -- enforced at compile time.
        """
        if from_node in self._conditional_edges:
            raise WorkflowDefinitionError(f"node {from_node!r} already has a conditional outgoing edge")
        self._conditional_edges[from_node] = _ConditionalEdge(from_node=from_node, router=router, edge_map=edge_map)
        self._compiled = None

    def set_entry(self, name: str) -> None:
        """Set the workflow's entry node. Exactly one entry is required."""
        self._entry = name
        self._compiled = None

    def set_exit(self, name: str) -> None:
        """Register an exit node. Multiple calls register multiple exits.

        When a registered exit node completes, its branch terminates;
        the workflow itself terminates when no nodes remain active.
        If ``set_exit`` is never called, all sink nodes (nodes with no
        outgoing edges) are treated as exits implicitly.
        """
        self._exits.add(name)
        self._compiled = None

    # -- Layer C: decorator API --------------------------------------------

    def node(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Callable[..., Any]:
        """Decorator: register a function as a Layer-C node.

        Two usage forms:

        ``@flow.node`` -- bare decorator. Node name derived from
        ``fn.__name__``.

        ``@flow.node(name="...", timeout=10.0)`` -- decorator factory
        with overrides.

        Layer-C semantics:

        - Parameter names bind from state at run time.
        - Return value lands in state under the node's name.
        - Parameter names that match other registered node names
          become static-edge dependencies (resolved at compile time).
        - Parameter names that don't match any node are workflow
          inputs and must be supplied via ``run(**initial_state)``.

        The decorator returns the original function unchanged, so a
        decorated function remains callable for unit tests independently
        of the workflow.

        Example::

            flow = Workflow()

            @flow.node
            def search(query: str) -> list[str]:
                return rag.search(query)

            @flow.node(name="summarize", timeout=30.0)
            def make_summary(search: list[str]) -> str:
                return llm(...).text

            flow.set_entry("search")
            flow.set_exit("summarize")
            result = flow.run(query="...")
        """

        def decorator(user_fn: Callable[..., Any]) -> Callable[..., Any]:
            node_name = name or getattr(user_fn, "__name__", None)
            if not node_name:
                raise WorkflowDefinitionError("@flow.node target has no derivable __name__; pass name= explicitly")
            self._add_layer_c_node(node_name, user_fn, timeout=timeout)
            return user_fn

        if fn is not None:
            # Bare @flow.node form: fn is the decorated function.
            return decorator(fn)
        # Parameterized form: @flow.node(name=..., timeout=...).
        return decorator

    def route(self, after: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register a function as a Layer-C conditional router.

        The decorated function takes named parameters bound from state
        (just like ``@flow.node``) and returns either a target node name
        (string) or the :data:`END` sentinel to terminate the branch.

        ``after`` is the name of the node whose completion triggers the
        router. A node may have at most one router.

        Desugars to :meth:`add_conditional_edge` with ``edge_map=None``.

        Example::

            @flow.route(after="search")
            def route_after_search(search: list[str]) -> str:
                return "summarize" if search else "fallback"
        """
        if not after:
            raise WorkflowDefinitionError("@flow.route(after=...) requires a non-empty target")

        def decorator(user_fn: Callable[..., Any]) -> Callable[..., Any]:
            param_names = _extract_param_names(user_fn, role="router")
            wrapper = _layer_c_router_wrapper(user_fn, param_names)
            self.add_conditional_edge(after, wrapper, edge_map=None)
            self._layer_c_router_meta[after] = _LayerCRouterMeta(
                fn_name=getattr(user_fn, "__name__", "<anon>"),
                param_names=tuple(param_names),
            )
            return user_fn

        return decorator

    # -- Inspection --------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, _Node]:
        """Read-only view of registered nodes (defensive copy)."""
        return dict(self._nodes)

    @property
    def static_edges(self) -> Dict[str, List[str]]:
        """Read-only view of static edges (defensive copy)."""
        return {k: list(v) for k, v in self._static_edges.items()}

    @property
    def conditional_edges(self) -> Dict[str, _ConditionalEdge]:
        """Read-only view of conditional edges (defensive copy)."""
        return dict(self._conditional_edges)

    @property
    def state_schema(self) -> Optional[type]:
        return self._state_schema

    @property
    def derived_state_schema(self) -> Dict[str, Any]:
        """Synthesize a state-shape dict from Layer-C metadata.

        Returns a mapping from state key -> declared type (or ``Any``
        when unannotated). Keys include:

        - One entry per Layer-C node, whose value is the node's return
          annotation (or ``Any``).
        - One entry per "workflow input" -- a Layer-C parameter name
          that didn't match any other node. Type defaults to ``Any``
          (we don't introspect across functions to infer it).

        This is purely informational -- Phase 1/2 doesn't enforce
        schema agreement. Phase 4 will use this for typed-state
        compilation. Useful now for ``flow.derived_state_schema``
        debugging and for the future ``Workflow[StateT]`` generic.

        For workflows with an explicit ``state_schema``, returns that
        schema's annotations (when introspectable); otherwise the
        Layer-C-derived shape.
        """
        # If the user supplied an explicit schema, prefer it.
        if self._state_schema is not None:
            import typing as _t

            try:
                hints = _t.get_type_hints(self._state_schema)
                if hints:
                    return hints
            except Exception:
                pass
            ann = getattr(self._state_schema, "__annotations__", None)
            if ann is not None:
                return dict(ann)

        shape: Dict[str, Any] = {}
        # Each Layer-C node produces a state key under its name.
        for c_name, meta in self._layer_c_meta.items():
            ret = meta.return_annotation
            shape[c_name] = ret if ret is not None else Any
        # Each input requirement is a state key too.
        input_names: Set[str] = set()
        for meta in self._layer_c_meta.values():
            for param in meta.param_names:
                if param not in self._nodes:
                    input_names.add(param)
        for router_meta in self._layer_c_router_meta.values():
            for param in router_meta.param_names:
                if param not in self._nodes:
                    input_names.add(param)
        for name in input_names:
            if name not in shape:
                shape[name] = Any
        return shape

    @property
    def entry(self) -> Optional[str]:
        return self._entry

    @property
    def exits(self) -> Set[str]:
        return set(self._exits)

    # -- Compile + run -----------------------------------------------------

    def compile(self) -> "CompiledWorkflow[StateT]":
        """Validate the workflow and produce a :class:`CompiledWorkflow`.

        Validation:

        - An entry node is set.
        - All referenced names (in edges, conditional edges,
          ``edge_map`` values, entry, exits) are registered nodes.
        - No node has both static outgoing and conditional outgoing
          edges (ambiguous routing).
        - The static-edge subgraph is a DAG (Kahn's algorithm
          succeeds). Conditional edges can create cycles; those are
          bounded at runtime by ``max_steps``.

        Idempotent: re-compiling a definition that hasn't changed
        returns the cached :class:`CompiledWorkflow`. Any mutation
        (``add_*``, ``set_*``) invalidates the cache.

        Raises:
            WorkflowDefinitionError: On any validation failure.
        """
        if self._compiled is not None:
            return self._compiled

        # 0. Layer-C edge inference: for each Layer-C node, for each
        #    parameter whose name matches another registered node, add
        #    a static edge from that node. Self-references are ignored.
        #    Parameters that don't match any node are recorded as
        #    workflow inputs (validated against initial_state at run).
        layer_c_inputs: Set[str] = set()
        for c_name, meta in self._layer_c_meta.items():
            for param in meta.param_names:
                if param == c_name:
                    # Self-reference: would create a self-loop. Treat as
                    # a workflow input instead (the user probably meant
                    # to receive an initial value with the same name as
                    # the node).
                    layer_c_inputs.add(param)
                    continue
                if param in self._nodes:
                    # Add a static edge if not already present.
                    existing = self._static_edges.setdefault(param, [])
                    if c_name not in existing:
                        existing.append(c_name)
                        self._static_reverse.setdefault(c_name, []).append(param)
                else:
                    layer_c_inputs.add(param)
        # Router params: only validate input requirements; the conditional
        # edge already wires routing. Routers see their from_node's state
        # transitively, so any param that matches a node is satisfied by
        # the time the router runs -- no additional edges needed.
        for from_node, router_meta in self._layer_c_router_meta.items():
            for param in router_meta.param_names:
                if param not in self._nodes:
                    layer_c_inputs.add(param)

        # 1. Entry must be set and registered.
        if self._entry is None:
            raise WorkflowDefinitionError("no entry node set (call set_entry)")
        if self._entry not in self._nodes:
            raise WorkflowDefinitionError(f"entry node {self._entry!r} not registered")

        # 2. All static-edge endpoints must be registered.
        for from_node, targets in self._static_edges.items():
            if from_node not in self._nodes:
                raise WorkflowDefinitionError(f"static edge from unknown node {from_node!r}")
            for t in targets:
                if t not in self._nodes:
                    raise WorkflowDefinitionError(f"static edge to unknown node {t!r}")

        # 3. All conditional-edge endpoints must be registered, and
        #    no node has both static outgoing and conditional outgoing.
        for from_node, ce in self._conditional_edges.items():
            if from_node not in self._nodes:
                raise WorkflowDefinitionError(f"conditional edge from unknown node {from_node!r}")
            if from_node in self._static_edges and self._static_edges[from_node]:
                raise WorkflowDefinitionError(
                    f"node {from_node!r} has both static and conditional outgoing edges; pick one routing style"
                )
            if ce.edge_map is not None:
                for k, v in ce.edge_map.items():
                    if v is END:
                        continue
                    if not isinstance(v, str):
                        raise WorkflowDefinitionError(
                            f"conditional edge map value for {from_node!r} "
                            f"under key {k!r} must be a node name or END "
                            f"(got {type(v).__name__})"
                        )
                    if v not in self._nodes:
                        raise WorkflowDefinitionError(
                            f"conditional edge from {from_node!r} maps key {k!r} to unknown node {v!r}"
                        )

        # 4. All exits must be registered.
        for exit_name in self._exits:
            if exit_name not in self._nodes:
                raise WorkflowDefinitionError(f"exit node {exit_name!r} not registered")

        # 5. Topological sort over static edges (cycle detection).
        levels = self._compute_levels()

        # 6. If no explicit exits, derive from sinks.
        if self._exits:
            exits = frozenset(self._exits)
        else:
            exits = frozenset(
                n for n in self._nodes if not self._static_edges.get(n) and n not in self._conditional_edges
            )

        # Phase 5: pick the implicit answer_key when the user didn't
        # supply one. A single exit is the natural answer; otherwise
        # we leave it None and WorkflowResult.answer falls back to the
        # last-run node's state entry.
        answer_key = self._answer_key
        if answer_key is None and len(exits) == 1:
            (answer_key,) = exits

        self._compiled = CompiledWorkflow(
            nodes=dict(self._nodes),
            static_edges={k: tuple(v) for k, v in self._static_edges.items()},
            static_reverse={k: tuple(v) for k, v in self._static_reverse.items()},
            conditional_edges=dict(self._conditional_edges),
            entry=self._entry,
            exits=exits,
            state_schema=self._state_schema,
            max_steps=self._max_steps,
            levels=tuple(tuple(lvl) for lvl in levels),
            layer_c_inputs=frozenset(layer_c_inputs),
            reducers=dict(self._reducers),
            invariants=tuple(self._invariants),
            policy=self._policy,
            task_param=self._task_param,
            answer_key=answer_key,
        )
        return self._compiled

    def _compute_levels(self) -> List[List[str]]:
        """Topological sort over static edges using Kahn's algorithm.

        Returns levels (lists of node names) in topological order.
        Nodes within a level have no static-edge dependencies on each
        other and may run concurrently.

        Raises:
            WorkflowDefinitionError: If a cycle exists in static edges.
        """
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for from_node, targets in self._static_edges.items():
            for t in targets:
                in_degree[t] = in_degree.get(t, 0) + 1

        levels: List[List[str]] = []
        remaining = set(self._nodes.keys())

        while remaining:
            current = sorted(n for n in remaining if in_degree[n] == 0)
            if not current:
                cycle_nodes = sorted(remaining)
                raise WorkflowDefinitionError(f"cycle detected in static edges; nodes still in cycle: {cycle_nodes}")
            levels.append(current)
            for n in current:
                remaining.remove(n)
                for t in self._static_edges.get(n, []):
                    in_degree[t] -= 1

        return levels

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Metrics from the most recent ``run`` / ``stream`` call, or
        ``None`` if the workflow hasn't run yet.

        Useful for direct inspection; the :meth:`as_agent` adapter
        also delegates here for its own ``metrics`` property. Reads
        the cached :class:`CompiledWorkflow` since the compiled
        instance is what actually runs nodes; either path
        (``Workflow.run`` -> ``CompiledWorkflow.arun``, or direct
        ``CompiledWorkflow.run``) updates the same cache.
        """
        if self._compiled is None:
            return self._last_metrics
        return self._compiled.metrics or self._last_metrics

    def run(self, **initial_state: Any) -> WorkflowResult[StateT]:
        """Compile (if needed) and run the workflow synchronously.

        Canonical workflow API: kwargs become initial state.

        Returns:
            :class:`WorkflowResult` carrying the final state, success
            flag, metrics, and the full event log.

        For the AgentProtocol shape (``agent.run(task: str) ->
        AgentResult``), use :meth:`as_agent` to get an adapter:
        ``flow.as_agent().run("hello")``.
        """
        return asyncio.run(self.arun(**initial_state))

    async def arun(self, **initial_state: Any) -> WorkflowResult[StateT]:
        """Async variant of :meth:`run`."""
        compiled = self.compile()
        result = await compiled.arun(dict(initial_state))
        self._last_metrics = result.metrics
        return result

    def stream(self, **initial_state: Any) -> "_SyncEventIterator":
        """Compile (if needed) and stream workflow events synchronously.

        Yields :class:`AgentEvent` objects in execution order. See
        :meth:`CompiledWorkflow.astream` for the event sequence.
        """
        compiled = self.compile()
        return compiled.stream(dict(initial_state))

    async def astream(self, **initial_state: Any) -> "AsyncIterator[AgentEvent]":
        """Async variant of :meth:`stream`."""
        compiled = self.compile()
        async for event in compiled.astream(dict(initial_state)):
            if event.type == EventType.WORKFLOW_END:
                self._last_metrics = event.metadata.get("metrics")
            yield event

    def as_agent(
        self,
        *,
        task_param: Optional[str] = None,
        answer_key: Optional[str] = None,
    ) -> "_WorkflowAgentAdapter[StateT]":
        """Return an :class:`AgentProtocol`-conformant adapter.

        A workflow's native shape is "typed state in, typed state out";
        the AgentProtocol shape (string in, string out) is a lossy
        projection of that. This adapter makes the projection explicit
        at the call site -- callers that want to plug a workflow into
        :func:`agent_as_tool`, :class:`TieredAgentTeam`,
        :class:`ReflectionLoop`, etc. opt in via ``flow.as_agent()``.

        The adapter exposes ``run(task: str) -> AgentResult``,
        ``arun(task: str)``, ``stream(task: str)`` yielding
        :class:`AgentEvent`, ``astream(task: str)``, and ``metrics``.

        Args:
            task_param: Override the workflow's configured
                ``task_param`` for this adapter (useful when exposing
                the same workflow under multiple protocol facades).
                Defaults to the value set at construction.
            answer_key: Override the workflow's configured
                ``answer_key``.

        Returns:
            A :class:`_WorkflowAgentAdapter` bound to this workflow's
            compiled form.
        """
        return _WorkflowAgentAdapter(
            workflow=self.compile(),
            task_param=task_param if task_param is not None else self._task_param,
            answer_key=answer_key if answer_key is not None else self._answer_key,
        )

    def to_mermaid(self) -> str:
        """Return a Mermaid ``graph TD`` rendering of the compiled workflow."""
        return self.compile().to_mermaid()

    def to_dot(self) -> str:
        """Return a Graphviz ``digraph`` rendering of the compiled workflow."""
        return self.compile().to_dot()

    def dry_run(self) -> "DryRunPlan":
        """Return the static execution plan; see :meth:`CompiledWorkflow.dry_run`."""
        return self.compile().dry_run()


# ---------------------------------------------------------------------------
# CompiledWorkflow (runtime)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledWorkflow(Generic[StateT]):
    """Validated, runnable workflow. Returned by :meth:`Workflow.compile`.

    Immutable -- to change the graph, mutate the source :class:`Workflow`
    and recompile.
    """

    nodes: Dict[str, _Node]
    # Static edges: from_node -> tuple of to_nodes
    static_edges: Dict[str, Tuple[str, ...]]
    # Reverse static edges: to_node -> tuple of from_nodes
    static_reverse: Dict[str, Tuple[str, ...]]
    # Conditional edges: from_node -> ConditionalEdge
    conditional_edges: Dict[str, _ConditionalEdge]
    entry: str
    exits: frozenset[str]
    state_schema: Optional[type]
    max_steps: int
    # Topological levels over static edges only. Conditional edges are
    # resolved at runtime.
    levels: Tuple[Tuple[str, ...], ...]
    # Layer-C metadata: parameter names that don't match any registered
    # node. These must be supplied via ``run(**initial_state)``; the
    # runtime validates this before execution.
    layer_c_inputs: frozenset[str] = frozenset()
    # Phase 4: reducer registry keyed by state key.
    reducers: Dict[str, Callable[[Any, Any], Any]] = field(default_factory=dict)
    # Phase 4: workflow-level invariants checked after each node completes.
    invariants: Tuple[WorkflowInvariant, ...] = ()
    # Phase 4: default policy used when an invariant's own policy is None.
    policy: ContractPolicy = ContractPolicy.ENFORCE
    # Phase 5: AgentProtocol-shape adapters. ``task_param`` is the state
    # key a positional ``run(task)`` call binds to; ``answer_key`` is the
    # state key whose final value is returned by ``WorkflowResult.answer``.
    task_param: str = "task"
    answer_key: Optional[str] = None

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Metrics from the most recent ``run`` / ``stream`` call.

        Cached on the (frozen) instance via ``object.__setattr__`` from
        the run methods; ``None`` until the workflow has been run.
        """
        return getattr(self, "_last_metrics_cache", None)

    def _record_metrics(self, metrics: Optional[AgentMetrics]) -> None:
        # Frozen dataclass: bypass the immutability for the metrics cache.
        object.__setattr__(self, "_last_metrics_cache", metrics)

    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> WorkflowResult[StateT]:
        """Run the workflow synchronously.

        Args:
            initial_state: Optional dict of initial state values
                forwarded to the entry node. Use kwargs on
                :meth:`Workflow.run` for the ergonomic form.

        For the AgentProtocol shape (``agent.run(task: str) ->
        AgentResult``), use :meth:`as_agent`.
        """
        return asyncio.run(self.arun(initial_state))

    async def arun(self, initial_state: Optional[Dict[str, Any]] = None) -> WorkflowResult[StateT]:
        """Run the workflow asynchronously, returning a final result.

        Internally consumes the same event-yielding generator
        :meth:`astream` exposes, so the two views are consistent --
        ``result.events`` is the exact sequence ``astream()`` would
        have yielded.

        See :meth:`astream` for the execution model.
        """
        events: List[AgentEvent] = []
        async for event in self._astream_events(initial_state):
            events.append(event)
        # The final WORKFLOW_END event carries the result snapshot.
        end = events[-1]
        meta = end.metadata
        result: WorkflowResult[StateT] = WorkflowResult(
            state=meta["state"],
            success=meta["success"],
            error=meta.get("error"),
            metrics=meta.get("metrics"),
            nodes_run=meta.get("nodes_run", []),
            events=events,
            answer_key=self.answer_key,
        )
        self._record_metrics(result.metrics)
        return result

    def stream(self, initial_state: Optional[Dict[str, Any]] = None) -> "_SyncEventIterator":
        """Synchronous event stream.

        Returns an iterator yielding :class:`AgentEvent` objects in
        the order they're emitted by the workflow runtime. The first
        event is always ``EventType.WORKFLOW_START``; the last is
        always ``EventType.WORKFLOW_END`` (whose metadata carries the
        final state, success flag, and error message if any).

        Internally drives an asyncio loop step by step so each yielded
        event is materialized as it occurs. For full async-native
        usage prefer :meth:`astream`.
        """

        async def _agen() -> AsyncIterator[AgentEvent]:
            async for ev in self._astream_events(initial_state):
                if ev.type == EventType.WORKFLOW_END:
                    self._record_metrics(ev.metadata.get("metrics"))
                yield ev

        return _SyncEventIterator(_agen())

    async def astream(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> "AsyncIterator[AgentEvent]":
        """Async event stream.

        Yields :class:`AgentEvent` objects as the workflow executes.
        Event sequence per run:

        - ``WORKFLOW_START`` with metadata ``{"entry": ..., "initial_state": ...}``.
        - One ``NODE_START`` per node about to run, with metadata
          ``{"node": name, "inputs": state_snapshot}``.
        - One ``NODE_END`` per node that completed (or ``ERROR`` if it
          raised), with metadata
          ``{"node": name, "update": dict, "elapsed_ms": float}``.
        - ``WORKFLOW_END`` with metadata
          ``{"state": dict, "success": bool, "error": str|None,
              "metrics": AgentMetrics, "nodes_run": list[str]}``.

        Nodes within the same topological level run concurrently;
        their ``NODE_START`` events all emit before the batch runs,
        and their ``NODE_END`` events all emit after the batch
        completes (in alphabetical order within the batch for
        determinism). Sub-agent events from agents wrapped via
        :func:`agent_node` or :func:`workflow_node` are forwarded
        through this stream with ``source`` set to the inner agent /
        workflow's name and ``parent_event_id`` linked to the
        ``NODE_START`` event id (Phase 5+).
        """
        async for event in self._astream_events(initial_state):
            if event.type == EventType.WORKFLOW_END:
                self._record_metrics(event.metadata.get("metrics"))
            yield event

    def as_agent(
        self,
        *,
        task_param: Optional[str] = None,
        answer_key: Optional[str] = None,
    ) -> "_WorkflowAgentAdapter[StateT]":
        """Return an :class:`AgentProtocol`-conformant adapter.

        Mirrors :meth:`Workflow.as_agent` but bound to this already-
        compiled instance.
        """
        return _WorkflowAgentAdapter(
            workflow=self,
            task_param=task_param if task_param is not None else self.task_param,
            answer_key=answer_key if answer_key is not None else self.answer_key,
        )

    async def _astream_events(self, initial_state: Optional[Dict[str, Any]] = None) -> "AsyncIterator[AgentEvent]":
        """Internal async generator: the canonical event source.

        Both :meth:`arun` and :meth:`astream` consume from this. The
        execution model documented on the old ``arun`` body lives here;
        the new outer methods are thin adaptors.
        """
        state: Dict[str, Any] = dict(initial_state or {})
        completed: Set[str] = set()
        active: Set[str] = {self.entry}
        nodes_run: List[str] = []
        metrics = AgentMetrics()
        start = time.perf_counter()
        steps = 0

        # Phase 4: track which node wrote each state key, so a second
        # writer without a registered reducer can be detected.
        # Initial-state keys are not tracked here; they're starting
        # points the user supplied, and a node writing under the same
        # name as an input is the normal pattern (initial input is
        # consumed; the node produces a new key under its own name).
        writers: Dict[str, str] = {}

        yield AgentEvent(
            type=EventType.WORKFLOW_START,
            content="",
            metadata={
                "entry": self.entry,
                "initial_state": dict(state),
            },
        )

        try:
            missing_inputs = self.layer_c_inputs - state.keys()
            if missing_inputs:
                raise WorkflowExecutionError(f"missing required workflow inputs: {sorted(missing_inputs)}")

            while active:
                if steps >= self.max_steps:
                    raise WorkflowExecutionError(
                        f"workflow exceeded max_steps ({self.max_steps}); "
                        f"completed: {sorted(completed)}, still active: {sorted(active)}"
                    )
                steps += 1

                ready = [n for n in sorted(active) if all(p in completed for p in self.static_reverse.get(n, ()))]
                if not ready:
                    raise WorkflowExecutionError(
                        f"workflow deadlocked: active nodes {sorted(active)} "
                        f"but no preds satisfied (completed: {sorted(completed)})"
                    )

                # Emit NODE_START for every ready node before kicking off
                # the batch -- consumers see the level structure clearly.
                node_start_times: Dict[str, float] = {}
                # Phase 5: stable id per node invocation so any forwarded
                # sub-events can point at it via ``parent_event_id``.
                node_event_ids: Dict[str, str] = {}
                for node_name in ready:
                    node_start_times[node_name] = time.perf_counter()
                    node_event_ids[node_name] = uuid.uuid4().hex
                    yield AgentEvent(
                        type=EventType.NODE_START,
                        content="",
                        metadata={
                            "node": node_name,
                            "inputs": dict(state),
                            "event_id": node_event_ids[node_name],
                        },
                    )

                # Set up real-time sub-event forwarding. Each node body
                # that supports streaming (``_StreamingAgentNodeBody`` /
                # ``_SubWorkflowNodeBody``) gets an ``_event_sink``
                # attribute pointing at a shared queue; the runtime
                # drains the queue concurrently with the node tasks so
                # sub-events appear in the outer stream in real time
                # rather than batched after each node completes.
                event_queue: "asyncio.Queue[Any]" = asyncio.Queue()
                event_loop = asyncio.get_running_loop()
                streaming_bodies: List[Any] = []
                for node_name in ready:
                    body = self.nodes[node_name].fn
                    if hasattr(body, "_event_sink"):
                        body._event_sink = (
                            event_loop,
                            event_queue,
                            node_name,
                            node_event_ids[node_name],
                        )
                        streaming_bodies.append(body)

                done_sentinel: object = object()

                async def _run_with_signal(name: str) -> Dict[str, Any]:
                    try:
                        return await self._run_one(name, state)
                    finally:
                        await event_queue.put(done_sentinel)

                tasks: Dict[str, "asyncio.Task[Dict[str, Any]]"] = {
                    name: asyncio.create_task(_run_with_signal(name)) for name in ready
                }

                # Drain the queue until every node has signalled completion.
                # Sub-events (forwarded by streaming node bodies) yield
                # immediately; the sentinel decrements the pending counter.
                pending_signals = len(ready)
                while pending_signals > 0:
                    item = await event_queue.get()
                    if item is done_sentinel:
                        pending_signals -= 1
                    else:
                        yield item

                # All tasks have completed (their `finally` ran). Collect
                # results; capture exceptions for the ERROR path below.
                results: List[Any] = []
                for node_name in ready:
                    try:
                        results.append(await tasks[node_name])
                    except BaseException as e:
                        results.append(e)

                # Clear the sink attribute so a re-run of the same node
                # (cycle case) doesn't carry stale state.
                for body in streaming_bodies:
                    body._event_sink = None

                # Process results in declaration order; emit NODE_END per
                # node (or ERROR + raise on the first failure).
                for node_name, outcome in zip(ready, results):
                    elapsed_ms = (time.perf_counter() - node_start_times[node_name]) * 1000.0
                    if isinstance(outcome, BaseException):
                        yield AgentEvent(
                            type=EventType.ERROR,
                            content=f"{type(outcome).__name__}: {outcome}",
                            metadata={"node": node_name, "elapsed_ms": elapsed_ms},
                        )
                        # Re-raise to terminate the workflow.
                        raise outcome
                    update = outcome

                    # Phase 4: apply reducers per-key and detect
                    # multi-writer collisions for keys without reducers.
                    for key, value in update.items():
                        if key in self.reducers:
                            state[key] = self.reducers[key](state.get(key), value)
                            # Reducer keys aren't tracked as single-writer;
                            # they're explicitly multi-writer.
                            writers[key] = "<reducer>"
                        elif key in writers and writers[key] != node_name:
                            raise WorkflowExecutionError(
                                f"state key {key!r} written by both "
                                f"{writers[key]!r} and {node_name!r}; "
                                f"register a reducer in Workflow(reducers={{...}}) "
                                f"if multi-writer is intended"
                            )
                        else:
                            state[key] = value
                            writers[key] = node_name

                    completed.add(node_name)
                    nodes_run.append(node_name)
                    metrics.tool_calls += 1
                    active.discard(node_name)

                    yield AgentEvent(
                        type=EventType.NODE_END,
                        content="",
                        metadata={
                            "node": node_name,
                            "update": dict(update),
                            "elapsed_ms": elapsed_ms,
                            "event_id": node_event_ids[node_name],
                        },
                    )

                    # Phase 4: invariant checks after each node completes.
                    # Yield CONTRACT_VIOLATION events; under ENFORCE,
                    # also raise to terminate the workflow.
                    if self.invariants and self.policy != ContractPolicy.IGNORE:
                        wf_state = WorkflowExecutionState(
                            state=dict(state),
                            elapsed_ms=(time.perf_counter() - start) * 1000.0,
                            node_count=len(nodes_run),
                            error_count=metrics.error_count,
                            estimated_cost_usd=float(state.get("estimated_cost_usd", 0.0)),
                            nodes_run=tuple(nodes_run),
                        )
                        for idx, inv in enumerate(self.invariants):
                            policy = inv.policy or self.policy
                            if policy == ContractPolicy.IGNORE:
                                continue
                            try:
                                ok = bool(inv.predicate(wf_state))
                            except Exception as e:
                                # A predicate that raises is itself a
                                # violation -- treat as the rule failing.
                                ok = False
                                inv_error: Optional[str] = f"predicate raised {type(e).__name__}: {e}"
                            else:
                                inv_error = None
                            if ok:
                                continue
                            violation = ContractViolation(
                                kind="workflow_invariant",
                                location=f"workflow#{idx}",
                                predicate=inv.message or f"invariant[{idx}]",
                                message=inv.message or f"workflow invariant {idx} failed",
                                context={
                                    "after_node": node_name,
                                    "predicate_error": inv_error,
                                    "elapsed_ms": wf_state.elapsed_ms,
                                    "node_count": wf_state.node_count,
                                },
                                policy=policy,
                            )
                            yield AgentEvent(
                                type=EventType.CONTRACT_VIOLATION,
                                content=str(violation),
                                metadata={
                                    "violation": violation,
                                    "index": idx,
                                    "after_node": node_name,
                                },
                            )
                            if policy in (
                                ContractPolicy.ENFORCE,
                                ContractPolicy.QUICK_ENFORCE,
                            ):
                                raise WorkflowExecutionError(
                                    f"workflow invariant violated (after {node_name!r}): {violation.message}"
                                )

                # Resolve outgoing edges for the just-completed nodes.
                for node_name in ready:
                    for succ in self.static_edges.get(node_name, ()):
                        if succ in completed:
                            continue
                        active.add(succ)
                    if node_name in self.conditional_edges:
                        ce = self.conditional_edges[node_name]
                        target = self._resolve_route(ce, state)
                        if target is END:
                            continue
                        assert isinstance(target, str)
                        if target in completed:
                            completed.discard(target)
                        active.add(target)

                if not active:
                    break

            metrics.total_time_ms = (time.perf_counter() - start) * 1000.0
            # Phase 5: emit a final ANSWER event so AgentProtocol-shape
            # consumers (ReflectionLoop, agent_as_tool, TieredAgentTeam)
            # see the canonical "I'm done" event. The content is the
            # ``answer_key`` value when set; otherwise the last-run
            # node's state entry. Falls back to "" when neither is
            # available.
            answer_value: str = ""
            if self.answer_key is not None and self.answer_key in state:
                answer_value = str(state[self.answer_key])
            elif nodes_run and nodes_run[-1] in state:
                answer_value = str(state[nodes_run[-1]])
            yield AgentEvent(
                type=EventType.ANSWER,
                content=answer_value,
                metadata={"answer_key": self.answer_key},
            )
            yield AgentEvent(
                type=EventType.WORKFLOW_END,
                content="",
                metadata={
                    "state": state,
                    "success": True,
                    "error": None,
                    "metrics": metrics,
                    "nodes_run": nodes_run,
                },
            )

        except (WorkflowExecutionError, WorkflowRoutingError) as e:
            metrics.total_time_ms = (time.perf_counter() - start) * 1000.0
            metrics.error_count += 1
            yield AgentEvent(
                type=EventType.WORKFLOW_END,
                content="",
                metadata={
                    "state": state,
                    "success": False,
                    "error": str(e),
                    "metrics": metrics,
                    "nodes_run": nodes_run,
                },
            )
        except BaseException as e:  # user node body raised
            metrics.total_time_ms = (time.perf_counter() - start) * 1000.0
            metrics.error_count += 1
            yield AgentEvent(
                type=EventType.WORKFLOW_END,
                content="",
                metadata={
                    "state": state,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "metrics": metrics,
                    "nodes_run": nodes_run,
                },
            )

    # -- Visualization + dry-run --------------------------------------------

    def to_mermaid(self) -> str:
        """Return a Mermaid ``graph TD`` representation of the workflow.

        Static edges render as solid arrows (``-->``); conditional
        edges render as dashed arrows with an edge-map key label
        (``-.->|"key"|``). Entry / exit nodes get distinct shapes so
        the diagram reads top-to-bottom by default.

        The output is a string; render via any Mermaid-compatible tool
        (mkdocs-mermaid, GitHub markdown, the mermaid.live editor).
        """
        lines: List[str] = ["graph TD"]
        # Node declarations: mark entry and exits with distinct shapes.
        for name in sorted(self.nodes):
            if name == self.entry and name in self.exits:
                lines.append(f"    {name}([{name}])")
            elif name == self.entry:
                lines.append(f"    {name}(({name}))")
            elif name in self.exits:
                lines.append(f"    {name}([{name}])")
            else:
                lines.append(f"    {name}[{name}]")
        # Static edges.
        for from_node in sorted(self.static_edges):
            for to_node in self.static_edges[from_node]:
                lines.append(f"    {from_node} --> {to_node}")
        # Conditional edges.
        for from_node in sorted(self.conditional_edges):
            ce = self.conditional_edges[from_node]
            if ce.edge_map:
                for key, target in ce.edge_map.items():
                    if target is END:
                        lines.append(f'    {from_node} -.->|"{key}"| END(((END)))')
                    else:
                        lines.append(f'    {from_node} -.->|"{key}"| {target}')
            else:
                # Without an edge_map, we don't know targets statically;
                # show a dashed self-annotation.
                lines.append(f'    {from_node} -.->|"(dynamic)"| ???')
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Return a Graphviz ``digraph`` representation of the workflow.

        Mirrors :meth:`to_mermaid` semantics in DOT syntax. Render via
        ``dot -Tpng workflow.dot``.
        """
        lines: List[str] = ["digraph workflow {", '    rankdir="TB";']
        # Node declarations.
        for name in sorted(self.nodes):
            attrs: List[str] = [f'label="{name}"']
            if name == self.entry:
                attrs.append("shape=circle")
            elif name in self.exits:
                attrs.append("shape=doublecircle")
            else:
                attrs.append("shape=box")
            lines.append(f"    {name} [{', '.join(attrs)}];")
        # Static edges.
        for from_node in sorted(self.static_edges):
            for to_node in self.static_edges[from_node]:
                lines.append(f"    {from_node} -> {to_node};")
        # Conditional edges (dashed).
        for from_node in sorted(self.conditional_edges):
            ce = self.conditional_edges[from_node]
            if ce.edge_map:
                for key, target in ce.edge_map.items():
                    if target is END:
                        lines.append(f'    {from_node} -> END [style=dashed, label="{key}"];')
                    else:
                        lines.append(f'    {from_node} -> {target} [style=dashed, label="{key}"];')
            else:
                lines.append(f'    {from_node} -> "?" [style=dashed, label="(dynamic)"];')
        lines.append("}")
        return "\n".join(lines)

    def dry_run(self) -> "DryRunPlan":
        """Return the static execution plan without running any nodes.

        Useful for inspecting workflow shape, validating the topological
        order, and surfacing which nodes are reachable only via
        conditional routing (and therefore not guaranteed to run).

        Conditional edges are not resolved -- they depend on runtime
        data the dry run doesn't have.
        """
        # Nodes that are reachable only through a conditional edge are
        # "conditional nodes". Static-edge targets of conditional edges
        # (when an edge_map is set) are included in this set.
        conditional_nodes: Set[str] = set()
        for ce in self.conditional_edges.values():
            if ce.edge_map is not None:
                for v in ce.edge_map.values():
                    if isinstance(v, str):
                        conditional_nodes.add(v)
            # Without an edge_map we can't statically determine targets.

        return DryRunPlan(
            levels=self.levels,
            conditional_nodes=frozenset(conditional_nodes),
            entry=self.entry,
            exits=self.exits,
            inputs_required=tuple(sorted(self.layer_c_inputs)),
        )

    async def _run_one(self, node_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node, returning its (possibly empty) state update.

        Sync functions are dispatched on :func:`asyncio.to_thread` so they
        don't block the event loop. Async functions are awaited directly.
        Per-node timeouts wrap the execution in :func:`asyncio.wait_for`.

        Raises whatever the node body raises; the caller's outer
        try/except handles capture into :class:`WorkflowResult`.
        """
        node = self.nodes[node_name]
        snapshot = dict(state)  # defensive copy so nodes can't mutate shared state

        async def _invoke() -> Dict[str, Any]:
            result = node.fn(snapshot)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return {}
            if not isinstance(result, dict):
                raise WorkflowExecutionError(
                    f"node {node_name!r} returned {type(result).__name__}, expected dict (or None)"
                )
            return result

        if node.fn is None:  # pragma: no cover -- defensive
            raise WorkflowExecutionError(f"node {node_name!r} has no callable")

        # Sync vs async dispatch. ``iscoroutinefunction(node.fn)``
        # misses callable instances whose ``__call__`` is async (e.g.
        # ``_SubWorkflowNodeBody``); check the bound method too.
        _is_async = asyncio.iscoroutinefunction(node.fn) or asyncio.iscoroutinefunction(
            getattr(node.fn, "__call__", None)
        )
        if _is_async:
            coro: Awaitable[Dict[str, Any]] = _invoke()
        else:
            # Sync callable: run on a worker thread to avoid blocking
            # the event loop. We wrap _invoke (which is async) in an
            # asyncio.to_thread of the *user* callable, then handle
            # the result-shape check here to keep _invoke's logic in
            # one place.
            async def _sync_invoke() -> Dict[str, Any]:
                result = await asyncio.to_thread(node.fn, snapshot)
                if result is None:
                    return {}
                if not isinstance(result, dict):
                    raise WorkflowExecutionError(
                        f"node {node_name!r} returned {type(result).__name__}, expected dict (or None)"
                    )
                return result

            coro = _sync_invoke()

        if node.timeout is not None:
            try:
                return await asyncio.wait_for(coro, timeout=node.timeout)
            except asyncio.TimeoutError as e:
                raise WorkflowExecutionError(f"node {node_name!r} exceeded timeout of {node.timeout}s") from e
        return await coro

    def _resolve_route(
        self,
        ce: _ConditionalEdge,
        state: Dict[str, Any],
    ) -> Union[str, _EndSentinel]:
        """Call the router and translate its return value into a node
        name (or :data:`END`).

        Raises:
            WorkflowRoutingError: If the router returns a value that
                isn't in ``edge_map`` (when ``edge_map`` is set), or
                returns an unknown node name (when ``edge_map`` is None).
        """
        snapshot = dict(state)
        try:
            raw = ce.router(snapshot)
        except Exception as e:
            raise WorkflowRoutingError(f"router for {ce.from_node!r} raised {type(e).__name__}: {e}") from e

        if raw is END:
            return cast(_EndSentinel, END)

        if ce.edge_map is not None:
            if raw not in ce.edge_map:
                raise WorkflowRoutingError(
                    f"router for {ce.from_node!r} returned {raw!r}, not in "
                    f"edge_map keys {sorted(ce.edge_map.keys(), key=str)}"
                )
            target = ce.edge_map[raw]
            if target is END:
                return END
            if target not in self.nodes:
                # Shouldn't happen post-compile, but guard anyway.
                raise WorkflowRoutingError(f"router for {ce.from_node!r} resolved to unknown target node {target!r}")
            return target

        # edge_map is None: raw is treated as the target name directly.
        if not isinstance(raw, str):
            raise WorkflowRoutingError(
                f"router for {ce.from_node!r} returned {type(raw).__name__} {raw!r}; expected a node name (str) or END"
            )
        if raw not in self.nodes:
            raise WorkflowRoutingError(f"router for {ce.from_node!r} returned {raw!r}, not a registered node")
        return raw


# ---------------------------------------------------------------------------
# Workflow-as-agent adapter
# ---------------------------------------------------------------------------


class _WorkflowAgentAdapter(Generic[StateT]):
    """Adapter exposing a :class:`CompiledWorkflow` via :class:`AgentProtocol`.

    Workflows have a native shape (typed state in, typed state out);
    the AgentProtocol shape (string in, string out) is a lossy
    projection. Constructed via :meth:`Workflow.as_agent` /
    :meth:`CompiledWorkflow.as_agent` so callers consciously opt in.

    The adapter:

    - ``run(task: str) -> AgentResult`` binds ``task`` to
      ``state[task_param]``, runs the inner workflow, and projects the
      final state to an :class:`AgentResult` (``answer`` = value at
      ``state[answer_key]`` stringified, falling back to the last-run
      node's state entry; ``steps`` = the full event log).
    - ``arun(task: str)`` is the async variant.
    - ``stream(task: str)`` / ``astream(task: str)`` yield
      :class:`AgentEvent` from the workflow's event stream.
    - ``metrics`` delegates to the underlying compiled workflow.

    Failure modes propagate cleanly: workflow runtime errors become
    ``AgentResult(success=False, error=...)`` rather than raising;
    only programming errors (e.g. ``task`` is not a ``str``) raise.
    """

    def __init__(
        self,
        workflow: "CompiledWorkflow[StateT]",
        task_param: str,
        answer_key: Optional[str],
    ) -> None:
        self._workflow = workflow
        self._task_param = task_param
        self._answer_key = answer_key

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Latest run's :class:`AgentMetrics` (delegates to the
        underlying compiled workflow)."""
        return self._workflow.metrics

    @property
    def workflow(self) -> "CompiledWorkflow[StateT]":
        """The wrapped compiled workflow (for callers that need it)."""
        return self._workflow

    @property
    def task_param(self) -> str:
        return self._task_param

    @property
    def answer_key(self) -> Optional[str]:
        return self._answer_key

    def _coerce_task(self, task: Any) -> Dict[str, Any]:
        if not isinstance(task, str):
            raise TypeError(f"_WorkflowAgentAdapter.run/stream: task must be a str (got {type(task).__name__})")
        return {self._task_param: task}

    def _project_result(self, result: WorkflowResult[StateT]) -> AgentResult:
        return AgentResult(
            answer=result.answer,
            steps=list(result.events),
            iterations=result.iterations,
            success=result.success,
            error=result.error,
            metrics=result.metrics,
        )

    def run(self, task: str) -> AgentResult:
        """Run the underlying workflow with ``state[task_param] = task``;
        project the result to an :class:`AgentResult`."""
        result = self._workflow.run(self._coerce_task(task))
        return self._project_result(result)

    async def arun(self, task: str) -> AgentResult:
        """Async variant of :meth:`run`."""
        result = await self._workflow.arun(self._coerce_task(task))
        return self._project_result(result)

    def stream(self, task: str) -> Iterator[AgentEvent]:
        """Yield :class:`AgentEvent` from the workflow's event stream."""
        yield from self._workflow.stream(self._coerce_task(task))

    async def astream(self, task: str) -> AsyncIterator[AgentEvent]:
        """Async variant of :meth:`stream`."""
        async for ev in self._workflow.astream(self._coerce_task(task)):
            yield ev


# ---------------------------------------------------------------------------
# Sync event iterator -- drives an async generator step-by-step on its own
# event loop. Exposed to sync callers via ``Workflow.stream()`` /
# ``CompiledWorkflow.stream()``.
# ---------------------------------------------------------------------------


class _SyncEventIterator:
    """Synchronous iterator over an async generator of :class:`AgentEvent`.

    Drives the underlying coroutine on a private event loop, materializing
    one event per ``__next__`` call. Created and returned from
    :meth:`Workflow.stream` / :meth:`CompiledWorkflow.stream`; users
    typically interact with it via ``for event in workflow.stream(...):``.

    Holds the event loop open for the iterator's lifetime. The loop is
    closed when iteration completes naturally (the async generator
    yields :class:`StopAsyncIteration`) or when the iterator is
    garbage-collected.
    """

    def __init__(self, agen: AsyncIterator[AgentEvent]) -> None:
        self._agen = agen
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False

    def __iter__(self) -> "Iterator[AgentEvent]":
        return self

    def __next__(self) -> AgentEvent:
        if self._closed:
            raise StopIteration
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        try:
            event = self._loop.run_until_complete(self._agen.__anext__())
        except StopAsyncIteration:
            self._close()
            raise StopIteration
        return event

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            try:
                # Drain any unfinished tasks (e.g., the async generator's
                # background work) before closing the loop.
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()
            self._loop = None

    def __del__(self) -> None:
        # Best-effort cleanup when the iterator goes out of scope mid-stream.
        try:
            self._close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers: wrap AgentProtocol / Tool as workflow nodes
# ---------------------------------------------------------------------------


class _StreamingAgentNodeBody:
    """Callable node body that streams a sub-agent's events.

    Used by :func:`agent_node` when ``forward_events=True``. While the
    sub-agent's stream is consumed (synchronously on a worker thread),
    each event is pushed in real time to the workflow runtime's event
    queue via :attr:`_event_sink` -- the outer stream observes
    sub-events as they happen rather than batched after the node
    completes. ``source`` and ``parent_event_id`` are stamped here so
    the runtime can yield the event unmodified.
    """

    # Set by the workflow runtime before invocation when the surrounding
    # workflow is streaming. Tuple of (event_loop, queue, source_name,
    # parent_event_id). When None, sub-events are not forwarded (callers
    # outside a workflow can still use the node body but won't see
    # real-time event propagation).
    _event_sink: Optional[Tuple[Any, Any, str, str]]

    def __init__(self, agent: Any, name: str, task_param: str) -> None:
        self._agent = agent
        self._name = name
        self._task_param = task_param
        self._event_sink = None
        self.__name__ = name

    def _publish(self, ev: AgentEvent) -> None:
        """Stamp source/parent_event_id and push the event to the sink.

        Called from a worker thread (this body runs via
        ``asyncio.to_thread``), so we use ``call_soon_threadsafe`` to
        hand the event to the event-loop-owned queue.
        """
        sink = self._event_sink
        if sink is None:
            return
        loop, queue, source, parent_id = sink
        if ev.source is None:
            ev.source = source
        if ev.parent_event_id is None:
            ev.parent_event_id = parent_id
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._task_param not in state:
            raise WorkflowExecutionError(f"agent_node({self._name!r}): state missing required key {self._task_param!r}")
        task = state[self._task_param]
        if not isinstance(task, str):
            raise WorkflowExecutionError(
                f"agent_node({self._name!r}): state[{self._task_param!r}] must be a str, got {type(task).__name__}"
            )

        # Consume the agent's event stream; the final ANSWER event's
        # content (or the last yielded event's content) becomes the
        # state update value. Each event is published to the sink as
        # soon as it's produced.
        answer = ""
        try:
            for ev in self._agent.stream(task):
                self._publish(ev)
                if ev.type == EventType.ANSWER:
                    answer = ev.content
        except Exception:
            # Stream-based path failed; fall back to ``run`` so the node
            # still produces an answer (and stream-incompatible agents
            # don't break the workflow). We surface the synchronous
            # ``run`` result as a single ANSWER event to the consumer.
            result = self._agent.run(task)
            answer = getattr(result, "answer", "") or ""
            self._publish(AgentEvent(type=EventType.ANSWER, content=answer))

        return {self._name: answer}


def agent_node(
    agent: Any,
    name: str,
    task_param: str = "task",
    *,
    forward_events: bool = True,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build a Layer-B workflow node that wraps an ``AgentProtocol``.

    The returned function takes the current workflow state, pulls the
    task string from ``state[task_param]``, runs the wrapped agent, and
    stores the answer under ``state[name]``. Errors from the agent
    propagate (the workflow's outer try/except captures them into
    :class:`WorkflowResult`).

    Args:
        agent: Any object satisfying :class:`AgentProtocol`
            (``ReActAgent``, ``ConstrainedAgent``, ``ContractAgent``,
            another :class:`Workflow` / :class:`CompiledWorkflow`, etc.).
            When the wrapped object is a workflow, :func:`workflow_node`
            is a more precise wrapper -- this function still works but
            doesn't propagate the inner workflow's typed state, only the
            scalar answer.
        name: Workflow-side node name; also the state key under which
            the agent's answer lands.
        task_param: Which state key the agent should receive as its
            task string. Default ``"task"``; rename when the workflow's
            state shape uses something more semantic.
        forward_events: When True (default), the wrapped agent is
            consumed via ``agent.stream(task)`` and the inner events are
            forwarded into the outer workflow's event stream between
            this node's ``NODE_START`` and ``NODE_END`` with ``source``
            and ``parent_event_id`` set. When False, the node uses the
            fast ``agent.run(task)`` path and emits only NODE_START /
            NODE_END (no sub-event nesting).

    Returns:
        A node function (or a stateful callable when ``forward_events``)
        with signature ``(state: dict) -> dict`` that can be registered
        via :meth:`Workflow.add_node`.

    Example::

        from cyllama.agents import ReActAgent
        from cyllama.agents.workflow import Workflow, agent_node

        researcher = ReActAgent(llm=llm, tools=[...])
        flow = Workflow()
        flow.add_node("research", agent_node(researcher, "research", task_param="topic"))
        flow.set_entry("research")
        result = flow.run(topic="quicksort partitioning strategies")
    """
    if forward_events:
        return _StreamingAgentNodeBody(agent, name, task_param)

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        if task_param not in state:
            raise WorkflowExecutionError(f"agent_node({name!r}): state missing required key {task_param!r}")
        task = state[task_param]
        if not isinstance(task, str):
            raise WorkflowExecutionError(
                f"agent_node({name!r}): state[{task_param!r}] must be a str, got {type(task).__name__}"
            )
        result = agent.run(task)
        # AgentResult has .answer; fall back to str(result) defensively.
        answer = getattr(result, "answer", None)
        if answer is None:
            answer = str(result)
        return {name: answer}

    _node.__name__ = name
    return _node


class _SubWorkflowNodeBody:
    """Callable node body that runs a sub-workflow and forwards events.

    Like :class:`_StreamingAgentNodeBody` but consumes
    ``CompiledWorkflow.astream`` (rather than ``AgentProtocol.stream``)
    so the inner workflow's structured state surface is preserved:
    the returned update merges the entire sub-state under the outer
    node's name when ``project_state=True``, or just the inner
    ``answer_key`` value otherwise.

    Inner events are forwarded to the outer workflow's event sink in
    real time as they arrive from ``astream``, not batched at the end.
    """

    _event_sink: Optional[Tuple[Any, Any, str, str]]

    def __init__(
        self,
        compiled: "CompiledWorkflow[Any]",
        name: str,
        task_param: Optional[str],
        project_state: bool,
    ) -> None:
        self._compiled = compiled
        self._name = name
        self._task_param = task_param
        self._project_state = project_state
        self._event_sink = None
        self.__name__ = name

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Build the sub-workflow's initial state. If a task_param is
        # configured we mirror the AgentProtocol shape -- wrap the
        # task value as ``{task_param: value}``. Otherwise forward the
        # inner workflow's declared inputs from the outer state.
        if self._task_param is not None:
            if self._task_param not in state:
                raise WorkflowExecutionError(
                    f"workflow_node({self._name!r}): state missing required key {self._task_param!r}"
                )
            inner_state: Dict[str, Any] = {self._task_param: state[self._task_param]}
        else:
            inputs = self._compiled.layer_c_inputs
            if inputs:
                inner_state = {k: state[k] for k in inputs if k in state}
            else:
                inner_state = dict(state)

        sink = self._event_sink
        sub_state: Dict[str, Any] = {}
        success = True
        error: Optional[str] = None
        async for ev in self._compiled.astream(inner_state):
            # Real-time forward each inner event to the outer queue.
            # ``__call__`` runs on the event loop so we can ``await
            # queue.put`` directly (no thread-safe hop needed).
            if sink is not None:
                _, queue, source, parent_id = sink
                if ev.source is None:
                    ev.source = source
                if ev.parent_event_id is None:
                    ev.parent_event_id = parent_id
                await queue.put(ev)
            if ev.type == EventType.WORKFLOW_END:
                sub_state = ev.metadata.get("state", {}) or {}
                success = bool(ev.metadata.get("success", True))
                error = ev.metadata.get("error")

        if not success:
            raise WorkflowExecutionError(f"workflow_node({self._name!r}): inner workflow failed: {error}")

        if self._project_state:
            # Surface the entire sub-state under our node name so callers
            # can reach into ``state[name]["some_inner_key"]``.
            return {self._name: dict(sub_state)}
        # Default: store only the inner answer (str-shaped).
        key = self._compiled.answer_key
        if key is not None and key in sub_state:
            return {self._name: sub_state[key]}
        return {self._name: ""}


def workflow_node(
    compiled: Any,
    name: str,
    *,
    task_param: Optional[str] = "task",
    project_state: bool = False,
) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Wrap a :class:`CompiledWorkflow` (or :class:`Workflow`) as a
    node usable inside another workflow.

    The inner workflow is executed via its async event stream so its
    events flow into the outer workflow with ``source`` set to ``name``
    and ``parent_event_id`` linked to the outer ``NODE_START`` event.
    Inner workflow failures are re-raised as :class:`WorkflowExecutionError`
    on the outer workflow (which captures them into ``WorkflowResult``).

    Args:
        compiled: A :class:`CompiledWorkflow` or a :class:`Workflow`
            (auto-compiled at first call). Must declare an ``answer_key``
            (single exit, by default) when ``project_state=False``.
        name: Outer-workflow node name; also the state key under which
            the result lands.
        task_param: When set (default ``"task"``), the inner workflow is
            invoked AgentProtocol-style with ``state[task_param]`` as
            the positional task. Set to ``None`` to forward the inner
            workflow's declared inputs from the outer state instead.
        project_state: When True, the entire final sub-state dict lands
            under ``state[name]``. When False (default), only the inner
            ``answer_key`` value is surfaced (the AgentResult-shape
            answer).
    """
    if isinstance(compiled, Workflow):
        compiled = compiled.compile()
    if not isinstance(compiled, CompiledWorkflow):
        raise WorkflowDefinitionError(
            f"workflow_node: expected Workflow / CompiledWorkflow, got {type(compiled).__name__}"
        )
    return _SubWorkflowNodeBody(
        compiled=compiled,
        name=name,
        task_param=task_param,
        project_state=project_state,
    )


def tool_node(
    tool: Any,
    name: Optional[str] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build a Layer-B workflow node that wraps a :class:`Tool`.

    Reads tool parameters from state by name (matching the tool's
    declared JSON-schema ``properties``), invokes the tool, and stores
    the result under ``state[name or tool.name]``.

    Args:
        tool: A :class:`cyllama.agents.Tool` instance (or any object
            with ``name``, ``parameters``, and a callable interface).
        name: Workflow-side node name. Defaults to ``tool.name``.

    Returns:
        A node function with signature ``(state: dict) -> dict``.

    Example::

        from cyllama.agents import tool
        from cyllama.agents.workflow import Workflow, tool_node

        @tool
        def search(query: str) -> list[str]:
            return [...]

        flow = Workflow()
        flow.add_node("search", tool_node(search))
        flow.set_entry("search")
        result = flow.run(query="...")

    Errors from the tool body propagate; per-tool ``timeout`` (if set)
    is honored by the tool's own machinery -- the workflow doesn't
    second-guess it. Argument coercion (:func:`coerce_args`) is *not*
    applied here because the wrapper builds typed kwargs directly from
    state by name; the values are assumed to be the right type already
    (the upstream nodes that produced them are typed).
    """
    node_name = name or getattr(tool, "name", None)
    if not node_name:
        raise WorkflowDefinitionError("tool_node: tool has no .name attribute; pass name= explicitly")
    params = getattr(tool, "parameters", {}) or {}
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    param_names = list(properties.keys())

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for p in param_names:
            if p in state:
                kwargs[p] = state[p]
        result = tool(**kwargs)
        return {node_name: result}

    _node.__name__ = node_name
    return _node
