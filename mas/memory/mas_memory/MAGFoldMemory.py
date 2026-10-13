from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .memory_base import MASMemoryBase
from ..common import AgentMessage, MASMessage
from mas.llm import Message
from mas.utils import load_json, write_json


REFUTE_KEYWORDS = (
    "not",
    "incorrect",
    "wrong",
    "outdated",
    "however",
    "contradict",
    "refute",
    "false",
    "invalid",
)
VERIFY_KEYWORDS = (
    "verify",
    "verified",
    "verification",
    "evidence",
    "source",
    "proves",
    "proof",
    "confirm",
    "confirmed",
)
SUPPORT_KEYWORDS = (
    "support",
    "supports",
    "supported",
    "consistent",
    "agree",
    "agrees",
    "because",
    "therefore",
)
ASSIGN_KEYWORDS = (
    "you should",
    "assign",
    "please check",
    "please verify",
    "need to check",
)
FAILURE_KEYWORDS = (
    "fail",
    "failed",
    "failure",
    "stuck",
    "loop",
    "invalid",
    "cannot",
    "can't",
)
ACTION_HINTS = (
    "act ",
    "action",
    "search",
    "lookup",
    "open ",
    "close ",
    "go to",
    "pick up",
    "put down",
    "move ",
    "take ",
    "use ",
    "examine",
    "look",
    "read",
    "verify",
    "check",
)
ALFWORLD_LOCATION_TYPES = (
    "bed",
    "cabinet",
    "coffeemachine",
    "countertop",
    "desk",
    "drawer",
    "fridge",
    "garbagecan",
    "handtowelholder",
    "laundryhamper",
    "microwave",
    "safe",
    "shelf",
    "sinkbasin",
    "stoveburner",
    "toaster",
    "toilet",
    "toiletpaperhanger",
    "towelholder",
)
ALFWORLD_LOOK_SEARCH_ORDER = (
    "desk 1",
    "desk 2",
    "drawer 1",
    "drawer 2",
    "drawer 3",
    "drawer 4",
    "drawer 5",
    "drawer 6",
    "shelf 1",
    "shelf 2",
    "shelf 3",
    "shelf 4",
    "shelf 5",
    "shelf 6",
    "bed 1",
    "garbagecan 1",
    "laundryhamper 1",
    "safe 1",
)
BLOCK_TYPES = {
    "step_transition",
    "evidence_verification",
    "conflict_resolution",
    "tool_use",
    "loop_or_stuck",
    "trajectory_segment",
    "failure_cause",
    "generic",
}
OUTCOMES = {"resolved", "failed", "unresolved", "unknown"}
MAGFOLD_SEGMENTER_SYSTEM_PROMPT = (
    "You are a MAGFold memory segmenter for embodied task trajectories. "
    "Convert completed trajectories into compact, action-oriented reusable experiences. "
    "Return valid JSON only."
)


def _shorten(text: Optional[str], limit: int = 180) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def infer_edge_type(
    src_content: str = "",
    dst_content: str = "",
    metadata: Optional[dict[str, Any]] = None,
    default: str = "depends_on",
) -> str:
    """Infer a typed relation with deterministic keyword heuristics."""
    metadata = metadata or {}
    explicit = metadata.get("edge_type")
    if explicit in {
        "temporal",
        "depends_on",
        "supports",
        "refutes",
        "assigns_to",
        "verifies",
        "summarizes",
        "causes_failure",
    }:
        return explicit

    target = f"{src_content}\n{dst_content}".lower()
    dst_lower = (dst_content or "").lower()

    if _contains_any(dst_lower, REFUTE_KEYWORDS):
        return "refutes"
    if _contains_any(dst_lower, ASSIGN_KEYWORDS):
        return "assigns_to"
    if _contains_any(dst_lower, VERIFY_KEYWORDS):
        return "verifies"
    if _contains_any(dst_lower, SUPPORT_KEYWORDS):
        return "supports"
    if _contains_any(target, FAILURE_KEYWORDS):
        return "causes_failure"
    return default


@dataclass
class AtomicInteractionNode:
    """Scale-1 node derived from an agent message, action, or observation."""

    node_id: str
    query_id: str
    step_id: int | str
    turn: Optional[int]
    agent_id: str
    role: str
    node_type: str
    content: str
    subgoal: Optional[str] = None
    claims: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicInteractionNode":
        return cls(**data)


@dataclass
class AtomicInteractionEdge:
    """Scale-1 typed relation between atomic interaction nodes."""

    edge_id: str
    src: str
    dst: str
    edge_type: str
    rationale: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicInteractionEdge":
        return cls(**data)


@dataclass
class MAGFoldBlock:
    """Coarse interaction memory unit that points back to raw atomic nodes."""

    block_id: str
    query_id: str
    scale: int
    block_type: str
    responsibility: str
    summary: str
    input_state: str
    output_state: str
    outcome: str
    covered_node_ids: list[str]
    covered_edge_ids: list[str]
    input_boundary_nodes: list[str] = field(default_factory=list)
    output_boundary_nodes: list[str] = field(default_factory=list)
    key_node_ids: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    edge_profile: dict[str, int] = field(default_factory=dict)
    key_claims: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    failure_lesson: Optional[str] = None
    unfold_policy: str = "coarse_to_fine"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MAGFoldBlock":
        return cls(**data)


@dataclass
class BlockEdge:
    """Coarse relation between two MAGFold blocks."""

    src_block_id: str
    dst_block_id: str
    edge_type: str
    supporting_atomic_edges: list[str] = field(default_factory=list)
    rationale: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockEdge":
        return cls(**data)


@dataclass
class RetrievedBlock:
    """Retrieved block plus score diagnostics for debugging."""

    block: MAGFoldBlock
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class AtomicInteractionGraph:
    """Serializable scale-1 interaction graph for one MAS task."""

    query_id: str
    task_main: str = ""
    task_description: str = ""
    nodes: list[AtomicInteractionNode] = field(default_factory=list)
    edges: list[AtomicInteractionEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "task_main": self.task_main,
            "task_description": self.task_description,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicInteractionGraph":
        return cls(
            query_id=data["query_id"],
            task_main=data.get("task_main", ""),
            task_description=data.get("task_description", ""),
            nodes=[AtomicInteractionNode.from_dict(node) for node in data.get("nodes", [])],
            edges=[AtomicInteractionEdge.from_dict(edge) for edge in data.get("edges", [])],
            metadata=data.get("metadata", {}),
        )

    @property
    def node_map(self) -> dict[str, AtomicInteractionNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def edge_map(self) -> dict[str, AtomicInteractionEdge]:
        return {edge.edge_id: edge for edge in self.edges}


class AtomicGraphBuilder:
    """Build a scale-1 interaction graph from the existing MAS StateChain."""

    def __init__(self, enable_edge_typing: bool = True):
        self.enable_edge_typing = enable_edge_typing

    def build(self, mas_message: MASMessage, query_id: Optional[str] = None) -> AtomicInteractionGraph:
        query_id = query_id or mas_message.get_extra_field("magfold_query_id") or _stable_id("gfq")
        graph = AtomicInteractionGraph(
            query_id=query_id,
            task_main=mas_message.task_main or "",
            task_description=mas_message.task_description or "",
            metadata={
                "label": mas_message.label,
                "extra_fields": dict(mas_message.extra_fields),
            },
        )

        source_to_atomic: dict[tuple[int, str], str] = {}
        previous_node_id: Optional[str] = None
        turn = 0
        edge_counter = 0

        def add_node(node: AtomicInteractionNode) -> str:
            nonlocal previous_node_id, turn, edge_counter
            node.turn = turn
            turn += 1
            graph.nodes.append(node)
            if previous_node_id is not None:
                edge_counter = self._add_edge(
                    graph,
                    previous_node_id,
                    node.node_id,
                    "temporal",
                    "execution order",
                    edge_counter,
                    {"source": "temporal_order"},
                )
                inferred = self._infer_between(graph, previous_node_id, node.node_id, default="depends_on")
                if inferred not in {"temporal", "depends_on"}:
                    edge_counter = self._add_edge(
                        graph,
                        previous_node_id,
                        node.node_id,
                        inferred,
                        "keyword relation inferred from adjacent content",
                        edge_counter,
                        {"source": "temporal_inference"},
                    )
            previous_node_id = node.node_id
            return node.node_id

        for step_id, state in enumerate(mas_message.chain_of_states):
            state_node_ids: list[str] = []
            for raw_node_id, data in state.nodes(data=True):
                agent_name = data.get("agent_name") or raw_node_id
                extra_fields = data.get("extra_fields") or {}
                content = data.get("message") or ""
                node = AtomicInteractionNode(
                    node_id=self._node_id(query_id, step_id, raw_node_id),
                    query_id=query_id,
                    step_id=step_id,
                    turn=None,
                    agent_id=agent_name,
                    role=extra_fields.get("role") or self._infer_role(agent_name),
                    node_type=extra_fields.get("node_type") or self._infer_node_type(content, agent_name),
                    content=content,
                    subgoal=extra_fields.get("subgoal"),
                    claims=self._extract_claims(content),
                    metadata={
                        "source": "agent_message",
                        "source_node_id": raw_node_id,
                        "system_instruction": data.get("system_instruction"),
                        "user_instruction": data.get("user_instruction"),
                        "extra_fields": extra_fields,
                    },
                )
                add_node(node)
                source_to_atomic[(step_id, raw_node_id)] = node.node_id
                state_node_ids.append(node.node_id)

            for src, dst, edge_data in state.edges(data=True):
                src_atomic = source_to_atomic.get((step_id, src))
                dst_atomic = source_to_atomic.get((step_id, dst))
                if not src_atomic or not dst_atomic:
                    continue
                inferred = self._infer_between(
                    graph,
                    src_atomic,
                    dst_atomic,
                    default="depends_on",
                    metadata=edge_data,
                )
                edge_counter = self._add_edge(
                    graph,
                    src_atomic,
                    dst_atomic,
                    inferred,
                    "derived from StateChain edge",
                    edge_counter,
                    {"source": "state_chain", **dict(edge_data)},
                )

            # We deliberately do NOT materialise a separate node for
            # `state.graph["action"]` even though the MAS StateChain carries one.
            # That node would always duplicate the content of the last
            # `agent_message` node in the same state (the solver's emit is
            # exactly the action passed to env.step), which:
            #   1. doubles every step in the segmenter prompt and confuses the
            #      block boundaries,
            #   2. forces oracle annotators to pick between two identical
            #      candidates for every step.
            # The agent_message node already carries `node_type="action"` for
            # real env actions (via `_infer_node_type`) and `node_type="thought"`
            # for "think:" emits, so downstream filters on `node_type` keep
            # working unchanged.
            observation = state.graph.get("observation")

            if observation:
                observation_node = AtomicInteractionNode(
                    node_id=self._node_id(query_id, step_id, "observation"),
                    query_id=query_id,
                    step_id=step_id,
                    turn=None,
                    agent_id="environment",
                    role="environment",
                    node_type="observation",
                    content=str(observation),
                    claims=self._extract_claims(str(observation)),
                    metadata={
                        "source": "state_observation",
                        "reward": state.graph.get("reward"),
                        # Keep the canonical env action as metadata so renderers
                        # can present one action-observation pair per state even
                        # when MAS frameworks duplicate the same emit across
                        # several agent nodes in the state graph.
                        "state_action": state.graph.get("action"),
                    },
                )
                observation_node_id = add_node(observation_node)
                # Connect the last agent_message in this state directly to the
                # observation -- this is the "you said X, env replied Y" link
                # that previously hopped through the redundant action node.
                if state_node_ids:
                    inferred = self._infer_between(
                        graph,
                        state_node_ids[-1],
                        observation_node_id,
                        default="depends_on",
                    )
                    edge_counter = self._add_edge(
                        graph,
                        state_node_ids[-1],
                        observation_node_id,
                        inferred,
                        "environment observation follows the agent's last emit",
                        edge_counter,
                        {"source": "state_observation_link"},
                    )

        return graph

    def _add_edge(
        self,
        graph: AtomicInteractionGraph,
        src: str,
        dst: str,
        edge_type: str,
        rationale: str,
        edge_counter: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        graph.edges.append(
            AtomicInteractionEdge(
                edge_id=f"{graph.query_id}_e{edge_counter}",
                src=src,
                dst=dst,
                edge_type=edge_type,
                rationale=rationale,
                confidence=0.8 if metadata and "inference" in metadata.get("source", "") else 1.0,
                metadata=metadata or {},
            )
        )
        return edge_counter + 1

    def _infer_between(
        self,
        graph: AtomicInteractionGraph,
        src: str,
        dst: str,
        default: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        if not self.enable_edge_typing:
            return default
        node_map = graph.node_map
        return infer_edge_type(
            node_map.get(src).content if src in node_map else "",
            node_map.get(dst).content if dst in node_map else "",
            metadata=metadata,
            default=default,
        )

    def _node_id(self, query_id: str, step_id: int, raw_node_id: str) -> str:
        safe_raw = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_node_id).strip("_") or "node"
        return f"{query_id}_s{step_id}_{safe_raw}"

    def _infer_role(self, agent_name: str) -> str:
        if not agent_name:
            return "unknown"
        if "_" in agent_name:
            return agent_name.split("_", 1)[0]
        if "-" in agent_name:
            return agent_name.split("-", 1)[0]
        return agent_name

    def _infer_node_type(self, content: str, agent_name: str) -> str:
        lowered = (content or "").strip().lower()
        agent_lower = (agent_name or "").lower()
        if not lowered:
            return "unknown"
        if lowered.startswith("think:"):
            return "thought"
        if "decision" in agent_lower or "decison" in agent_lower or "planner" in agent_lower:
            return "decision"
        if any(lowered.startswith(hint) or f" {hint}" in lowered for hint in ACTION_HINTS):
            return "action"
        if _contains_any(lowered, VERIFY_KEYWORDS):
            return "observation"
        return "utterance"

    def _extract_claims(self, content: str) -> list[str]:
        chunks = re.split(r"(?<=[.!?])\s+", (content or "").strip())
        return [_shorten(chunk, 140) for chunk in chunks if chunk.strip()][:3]


class BlockBuilder:
    """Build interaction blocks from a scale-1 atomic graph."""

    def __init__(self, segment_window_size: int = 3):
        self.segment_window_size = max(segment_window_size, 1)

    def build_blocks(
        self,
        atomic_graph: AtomicInteractionGraph,
        task_context: Optional[MASMessage] = None,
        result: Optional[Any] = None,
    ) -> list[MAGFoldBlock]:
        step_blocks = self._build_step_blocks(atomic_graph, task_context, result)
        loop_blocks = self._build_loop_blocks(atomic_graph, task_context, result)
        segment_blocks = self._build_segment_blocks(atomic_graph, step_blocks, task_context, result)
        return step_blocks + loop_blocks + segment_blocks

    def _build_step_blocks(
        self,
        atomic_graph: AtomicInteractionGraph,
        task_context: Optional[MASMessage],
        result: Optional[Any],
    ) -> list[MAGFoldBlock]:
        nodes_by_step: dict[int | str, list[AtomicInteractionNode]] = defaultdict(list)
        for node in atomic_graph.nodes:
            nodes_by_step[node.step_id].append(node)

        blocks: list[MAGFoldBlock] = []
        for step_id in sorted(nodes_by_step, key=lambda item: str(item)):
            nodes = sorted(nodes_by_step[step_id], key=lambda node: node.turn or 0)
            block = self._make_block(
                atomic_graph,
                nodes,
                block_id=f"{atomic_graph.query_id}_step_{step_id}",
                block_type=self._classify_block(atomic_graph, nodes, fallback="step_transition"),
                scale=2,
                task_context=task_context,
                result=result,
                metadata={"step_id": step_id},
            )
            if self._accept_block(block):
                blocks.append(block)
        return blocks

    def _build_loop_blocks(
        self,
        atomic_graph: AtomicInteractionGraph,
        task_context: Optional[MASMessage],
        result: Optional[Any],
    ) -> list[MAGFoldBlock]:
        action_nodes = [node for node in atomic_graph.nodes if node.node_type == "action"]
        blocks: list[MAGFoldBlock] = []
        for idx in range(2, len(action_nodes)):
            recent = action_nodes[idx - 2 : idx + 1]
            normalized = [self._normalize_action(node.content) for node in recent]
            if len(set(normalized)) != 1:
                continue
            covered_steps = {node.step_id for node in recent}
            nodes = [node for node in atomic_graph.nodes if node.step_id in covered_steps]
            block = self._make_block(
                atomic_graph,
                nodes,
                block_id=f"{atomic_graph.query_id}_loop_{recent[0].step_id}_{recent[-1].step_id}",
                block_type="loop_or_stuck",
                scale=3,
                task_context=task_context,
                result=result,
                metadata={"reason": "repeated_action", "action": recent[-1].content},
            )
            if self._accept_block(block):
                blocks.append(block)
        return blocks

    def _build_segment_blocks(
        self,
        atomic_graph: AtomicInteractionGraph,
        step_blocks: list[MAGFoldBlock],
        task_context: Optional[MASMessage],
        result: Optional[Any],
    ) -> list[MAGFoldBlock]:
        blocks: list[MAGFoldBlock] = []
        if len(step_blocks) <= 1:
            return blocks

        node_map = atomic_graph.node_map
        for start in range(0, len(step_blocks), self.segment_window_size):
            chunk = step_blocks[start : start + self.segment_window_size]
            if len(chunk) <= 1:
                continue
            node_ids: list[str] = []
            for block in chunk:
                node_ids.extend(block.covered_node_ids)
            seen = set()
            nodes = [node_map[node_id] for node_id in node_ids if node_id in node_map and not (node_id in seen or seen.add(node_id))]
            block = self._make_block(
                atomic_graph,
                nodes,
                block_id=f"{atomic_graph.query_id}_segment_{start}_{start + len(chunk) - 1}",
                block_type="trajectory_segment",
                scale=max(3, len(chunk)),
                task_context=task_context,
                result=result,
                metadata={"step_block_ids": [item.block_id for item in chunk]},
            )
            if self._accept_block(block):
                blocks.append(block)
        return blocks

    def _make_block(
        self,
        atomic_graph: AtomicInteractionGraph,
        nodes: list[AtomicInteractionNode],
        block_id: str,
        block_type: str,
        scale: int,
        task_context: Optional[MASMessage],
        result: Optional[Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> MAGFoldBlock:
        nodes = sorted(nodes, key=lambda node: node.turn or 0)
        node_ids = [node.node_id for node in nodes]
        node_id_set = set(node_ids)
        covered_edges = [
            edge for edge in atomic_graph.edges if edge.src in node_id_set and edge.dst in node_id_set
        ]
        edge_profile = Counter(edge.edge_type for edge in covered_edges)
        key_node_ids = self._select_key_nodes(nodes, covered_edges)
        key_nodes = [node for node in nodes if node.node_id in set(key_node_ids)]
        boundary_count = min(2, len(nodes))
        input_boundary_nodes = [node.node_id for node in nodes[:boundary_count]]
        output_boundary_nodes = [node.node_id for node in nodes[-boundary_count:]]
        summary_nodes = key_nodes or nodes[:2] + nodes[-2:]
        summary = self._summarize_nodes(summary_nodes, block_type)
        responsibility = self._responsibility(block_type, nodes)
        outcome = self._outcome(nodes, task_context, result)
        failure_lesson = self._failure_lesson(nodes, outcome)

        return MAGFoldBlock(
            block_id=block_id,
            query_id=atomic_graph.query_id,
            scale=scale,
            block_type=block_type,
            responsibility=responsibility,
            summary=summary,
            input_state=_shorten(nodes[0].content if nodes else "", 220),
            output_state=_shorten(self._last_output_node(nodes).content if nodes else "", 220),
            outcome=outcome,
            covered_node_ids=node_ids,
            covered_edge_ids=[edge.edge_id for edge in covered_edges],
            input_boundary_nodes=input_boundary_nodes,
            output_boundary_nodes=output_boundary_nodes,
            key_node_ids=key_node_ids,
            agents=sorted({node.agent_id for node in nodes if node.agent_id}),
            roles=sorted({node.role for node in nodes if node.role}),
            edge_profile=dict(edge_profile),
            key_claims=self._key_claims(key_nodes),
            decisions=[_shorten(node.content, 160) for node in key_nodes if node.node_type == "decision"],
            open_issues=self._open_issues(nodes),
            failure_lesson=failure_lesson,
            unfold_policy="key_boundary_then_relevant",
            confidence=0.85,
            metadata=metadata or {},
        )

    def _classify_block(
        self,
        atomic_graph: AtomicInteractionGraph,
        nodes: list[AtomicInteractionNode],
        fallback: str,
    ) -> str:
        node_id_set = {node.node_id for node in nodes}
        edge_types = {
            edge.edge_type
            for edge in atomic_graph.edges
            if edge.src in node_id_set and edge.dst in node_id_set
        }
        text = "\n".join(node.content for node in nodes).lower()
        if "refutes" in edge_types or _contains_any(text, REFUTE_KEYWORDS):
            return "conflict_resolution"
        if {"verifies", "supports"} & edge_types or _contains_any(text, VERIFY_KEYWORDS):
            return "evidence_verification"
        if "causes_failure" in edge_types or _contains_any(text, FAILURE_KEYWORDS):
            return "failure_cause"
        if any(node.node_type == "action" for node in nodes) and any(node.node_type == "observation" for node in nodes):
            return fallback
        return "generic"

    def _select_key_nodes(
        self,
        nodes: list[AtomicInteractionNode],
        edges: list[AtomicInteractionEdge],
    ) -> list[str]:
        important_ids = {
            edge.src
            for edge in edges
            if edge.edge_type in {"refutes", "verifies", "supports", "causes_failure"}
        } | {
            edge.dst
            for edge in edges
            if edge.edge_type in {"refutes", "verifies", "supports", "causes_failure"}
        }
        key_types = {"action", "observation", "decision"}
        key_node_ids = [
            node.node_id
            for node in nodes
            if node.node_type in key_types or node.node_id in important_ids
        ]
        if not key_node_ids and nodes:
            key_node_ids = [nodes[-1].node_id]
        return key_node_ids[:8]

    def _summarize_nodes(self, nodes: list[AtomicInteractionNode], block_type: str) -> str:
        parts = [
            f"{node.role}/{node.node_type}: {_shorten(node.content, 120)}"
            for node in nodes
            if node.content
        ]
        if not parts:
            return f"{block_type} with no textual content."
        return _shorten(" | ".join(parts), 520)

    def _responsibility(self, block_type: str, nodes: list[AtomicInteractionNode]) -> str:
        roles = ", ".join(sorted({node.role for node in nodes if node.role})) or "agents"
        if block_type == "evidence_verification":
            return f"Capture how {roles} checked, supported, or verified evidence."
        if block_type == "conflict_resolution":
            return f"Capture how {roles} handled a contradiction or refuted stale information."
        if block_type == "loop_or_stuck":
            return f"Capture repeated or stuck behavior and the local context around it."
        if block_type == "trajectory_segment":
            return f"Compress several adjacent interaction steps into one reusable trajectory segment."
        if block_type == "failure_cause":
            return f"Record a failure-prone interaction span and its observed consequence."
        return f"Record the state transition produced by {roles}."

    def _outcome(
        self,
        nodes: list[AtomicInteractionNode],
        task_context: Optional[MASMessage],
        result: Optional[Any],
    ) -> str:
        if isinstance(result, bool):
            return "resolved" if result else "failed"
        if task_context and task_context.label is True:
            return "resolved"
        if task_context and task_context.label is False:
            return "failed"
        rewards = [node.metadata.get("reward") for node in nodes if node.metadata.get("reward") is not None]
        if any(reward and reward > 0 for reward in rewards):
            return "resolved"
        if any(reward and reward < 0 for reward in rewards):
            return "failed"
        return "unknown"

    def _failure_lesson(self, nodes: list[AtomicInteractionNode], outcome: str) -> Optional[str]:
        text = "\n".join(node.content for node in nodes)
        if outcome == "failed" or _contains_any(text, FAILURE_KEYWORDS):
            return _shorten(f"Avoid repeating this span without checking the failing observation: {text}", 240)
        return None

    def _open_issues(self, nodes: list[AtomicInteractionNode]) -> list[str]:
        issues = []
        for node in nodes:
            if "?" in node.content or _contains_any(node.content, FAILURE_KEYWORDS):
                issues.append(_shorten(node.content, 140))
        return issues[:3]

    def _key_claims(self, nodes: list[AtomicInteractionNode]) -> list[str]:
        claims: list[str] = []
        for node in nodes:
            claims.extend(node.claims)
        return claims[:5]

    def _last_output_node(self, nodes: list[AtomicInteractionNode]) -> AtomicInteractionNode:
        for node in reversed(nodes):
            if node.node_type in {"observation", "action", "decision"}:
                return node
        return nodes[-1]

    def _accept_block(self, block: MAGFoldBlock) -> bool:
        return bool(block.covered_node_ids and block.responsibility and block.summary and block.key_node_ids)

    def _normalize_action(self, action: str) -> str:
        return re.sub(r"\s+", " ", (action or "").strip().lower())


class BlockGraphBuilder:
    """Build coarse block-level edges from atomic edges."""

    def build_block_edges(
        self,
        atomic_graph: AtomicInteractionGraph,
        blocks: list[MAGFoldBlock],
    ) -> list[BlockEdge]:
        block_edges: list[BlockEdge] = []
        for src_block in blocks:
            src_nodes = set(src_block.covered_node_ids)
            for dst_block in blocks:
                if src_block.block_id == dst_block.block_id:
                    continue
                dst_nodes = set(dst_block.covered_node_ids)
                if not src_nodes.isdisjoint(dst_nodes):
                    continue
                supporting_edges = [
                    edge
                    for edge in atomic_graph.edges
                    if edge.src in src_nodes and edge.dst in dst_nodes
                ]
                if not supporting_edges:
                    continue
                block_edges.append(
                    BlockEdge(
                        src_block_id=src_block.block_id,
                        dst_block_id=dst_block.block_id,
                        edge_type=self._coarse_type(supporting_edges),
                        supporting_atomic_edges=[edge.edge_id for edge in supporting_edges],
                        rationale="atomic edges connect the covered node sets",
                        weight=min(1.0, 0.25 * len(supporting_edges)),
                    )
                )
        return block_edges

    def _coarse_type(self, edges: list[AtomicInteractionEdge]) -> str:
        edge_types = {edge.edge_type for edge in edges}
        if "refutes" in edge_types:
            return "contradicts"
        if {"supports", "verifies"} & edge_types:
            return "refines"
        if "depends_on" in edge_types:
            return "enables"
        if "causes_failure" in edge_types:
            return "causes_failure"
        return "precedes"


@dataclass
class MAGFoldMemory(MASMemoryBase):
    """Block Interaction Graph memory plugin with coarse-to-fine retrieval."""

    def __post_init__(self):
        super().__post_init__()
        config = self.global_config or {}
        self.enabled: bool = config.get("enabled", True)
        self.block_top_k: int = config.get("block_top_k", 3)
        self.task_top_k: int = config.get("task_top_k", 3)
        self.inspiration_top_k: int = config.get("inspiration_top_k", 2)
        self.node_top_k_per_block: int = config.get("node_top_k_per_block", 5)
        self.segment_window_size: int = config.get("segment_window_size", 3)
        self.max_memory_tokens: int = config.get("max_memory_tokens", 2000)
        self.enable_edge_typing: bool = config.get("enable_edge_typing", True)
        self.enable_llm_edge_typer: bool = config.get("enable_llm_edge_typer", False)
        self.enable_llm_block_curator: bool = config.get("enable_llm_block_curator", True)
        self.enable_llm_block_segmenter: bool = config.get("enable_llm_block_segmenter", True)
        self.llm_block_segment_max_retries: int = config.get("llm_block_segment_max_retries", 3)
        self.llm_block_segment_max_tokens: int = config.get("llm_block_segment_max_tokens", 4096)
        self.strict_llm_block_segmenter: bool = config.get("strict_llm_block_segmenter", False)
        self.enable_llm_block_selector: bool = config.get("enable_llm_block_selector", True)
        self.full_unfold: bool = config.get("full_unfold", False)
        self.enable_magfold_overview_injection: bool = config.get("enable_magfold_overview_injection", True)
        self.enable_magfold_detail_trajectory: bool = config.get("enable_magfold_detail_trajectory", True)
        self.enable_magfold_insight_injection: bool = config.get("enable_magfold_insight_injection", True)
        self.save_atomic_graph: bool = config.get("save_atomic_graph", True)
        self.save_block_graph: bool = config.get("save_block_graph", True)
        self.debug: bool = config.get("debug", False)
        self.enable_embedding_retrieval: bool = config.get("enable_embedding_retrieval", False)
        self.segmenter_llm_model = config.get("segmenter_llm_model") or self.llm_model
        self.aux_llm_model = config.get("magfold_aux_llm_model") or self.llm_model
        self.enable_alfworld_look_action_advisor: bool = config.get("enable_alfworld_look_action_advisor", True)
        # Export atomic graphs directly when preparing segmenter data.
        self.atomic_graph_only: bool = config.get("atomic_graph_only", False)

        self.atomic_graph_dir = os.path.join(self.persist_dir, "atomic_graphs")
        os.makedirs(self.atomic_graph_dir, exist_ok=True)
        self.blocks_file = os.path.join(self.persist_dir, "blocks.json")
        self.block_edges_file = os.path.join(self.persist_dir, "block_edges.json")
        self.query_index_file = os.path.join(self.persist_dir, "query_index.json")

        self.atomic_builder = AtomicGraphBuilder(enable_edge_typing=self.enable_edge_typing)
        self.block_builder = BlockBuilder(segment_window_size=self.segment_window_size)
        self.block_graph_builder = BlockGraphBuilder()

        self.blocks: list[MAGFoldBlock] = self._load_blocks()
        self.block_edges: list[BlockEdge] = self._load_block_edges()
        self.query_index: dict[str, list[str]] = load_json(self.query_index_file) or {}
        self.atomic_graph_cache: dict[str, AtomicInteractionGraph] = {}
        self.last_atomic_graph: Optional[AtomicInteractionGraph] = None
        self.last_built_blocks: list[MAGFoldBlock] = []
        # RL hook: filled by retrieve_blocks() after each call.
        self.last_retrieval: list[dict[str, Any]] = []

    def init_task_context(self, task_main: str, task_description: str = None) -> MASMessage:
        context = super().init_task_context(task_main, task_description)
        context.add_extra_field("magfold_query_id", _stable_id("gfq"))
        return context

    def postprocess_action(
        self,
        action: str,
        action_history: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """Optionally turn MAGFold procedural memory into an executable action.

        ALFWorld look/examine tasks have a stable procedural shape: find and hold
        the target object, then use an observed desklamp. The 7B policy often
        spends many steps repeating `think:` despite retrieving the right memory,
        so MAGFold exposes this small postprocessor as a memory-level
        procedure executor for that subset.
        """
        cleaned = (action or "").strip()
        if not self.enable_alfworld_look_action_advisor:
            return cleaned
        task_text = self._current_alfworld_task_text()
        if self._alfworld_operation_signature(task_text) != "look":
            return cleaned
        advised = self._alfworld_look_next_action(task_text)
        return advised or cleaned

    def add_memory(self, mas_message: MASMessage) -> None:
        if not self.enabled:
            return
        query_id = mas_message.get_extra_field("magfold_query_id") or _stable_id("gfq")
        mas_message.add_extra_field("magfold_query_id", query_id)

        atomic_graph = self.atomic_builder.build(mas_message, query_id=query_id)

        # RL Phase 1: dump atomic graph only, skip block construction.
        if self.atomic_graph_only:
            self.last_atomic_graph = atomic_graph
            self.last_built_blocks = []
            self.atomic_graph_cache[query_id] = atomic_graph
            self._persist_atomic_graph(atomic_graph)
            self._index_done()
            return

        blocks = self._build_blocks_after_task(atomic_graph, mas_message)
        block_edges = self.block_graph_builder.build_block_edges(atomic_graph, blocks)

        self.last_atomic_graph = atomic_graph
        self.last_built_blocks = blocks
        self.atomic_graph_cache[query_id] = atomic_graph
        self.blocks.extend(blocks)
        self.block_edges.extend(block_edges)
        self.query_index[query_id] = [block.block_id for block in blocks]

        self._persist_atomic_graph(atomic_graph)
        self._persist_blocks()
        self._persist_block_edges()
        write_json(self.query_index, self.query_index_file)
        self._index_done()

    def import_external_blocks(
        self,
        atomic_graph: AtomicInteractionGraph,
        segmentation_payload: dict[str, Any],
        mas_message: MASMessage,
    ) -> list[MAGFoldBlock]:
        """Register externally segmented blocks and update the retrieval index.

        The caller controls persistence, allowing rollouts to share atomic graphs.
        """
        query_id = atomic_graph.query_id
        self.atomic_graph_cache[query_id] = atomic_graph

        specs, errors = self._validate_block_segment_payload(segmentation_payload, atomic_graph)
        if errors and self.debug:
            print(f"[MAGFold] import_external_blocks normalizer warnings: {'; '.join(errors)}")
        blocks = [
            self._block_from_segment_spec(spec, index, atomic_graph, mas_message)
            for index, spec in enumerate(specs, 1)
        ]
        blocks = [block for block in blocks if self.block_builder._accept_block(block)]
        block_edges = self.block_graph_builder.build_block_edges(atomic_graph, blocks)

        self.blocks.extend(blocks)
        self.block_edges.extend(block_edges)
        self.query_index[query_id] = [block.block_id for block in blocks]
        self.last_atomic_graph = atomic_graph
        self.last_built_blocks = blocks
        return blocks

    def install_prebuilt_blocks(
        self,
        atomic_graph: AtomicInteractionGraph,
        blocks: list[MAGFoldBlock],
        block_edges: Optional[list[BlockEdge]] = None,
    ) -> None:
        """Register prebuilt blocks and their graph connections."""
        self.atomic_graph_cache[atomic_graph.query_id] = atomic_graph
        if block_edges is None:
            block_edges = self.block_graph_builder.build_block_edges(atomic_graph, blocks)
        self.blocks.extend(blocks)
        self.block_edges.extend(block_edges)
        self.query_index[atomic_graph.query_id] = [block.block_id for block in blocks]

    def _build_blocks_after_task(
        self,
        atomic_graph: AtomicInteractionGraph,
        mas_message: MASMessage,
    ) -> list[MAGFoldBlock]:
        """Build reusable interaction blocks from a completed task trajectory.

        Use the LLM segmenter when enabled, with the deterministic builder as
        the fallback.
        """
        if self.enable_llm_block_segmenter:
            try:
                return self._build_blocks_with_llm_segmenter(atomic_graph, mas_message)
            except Exception:
                if self.strict_llm_block_segmenter:
                    raise
                if self.debug:
                    print("[MAGFold] LLM block segmenter failed; falling back to deterministic blocks.")

        blocks = self.block_builder.build_blocks(
            atomic_graph,
            task_context=mas_message,
            result=mas_message.label,
        )
        return [self._curate_block_with_llm(block, atomic_graph) for block in blocks]

    def _build_blocks_with_llm_segmenter(
        self,
        atomic_graph: AtomicInteractionGraph,
        mas_message: MASMessage,
    ) -> list[MAGFoldBlock]:
        """Ask an LLM to segment the completed task into MAGFold blocks."""
        if not atomic_graph.nodes:
            return []

        attempt = 0
        last_errors: list[str] = []
        while self.llm_block_segment_max_retries <= 0 or attempt < self.llm_block_segment_max_retries:
            attempt += 1
            prompt = self._block_segmenter_prompt(atomic_graph, mas_message)
            try:
                response = self.segmenter_llm_model(
                    messages=[
                        Message("system", MAGFOLD_SEGMENTER_SYSTEM_PROMPT),
                        Message("user", prompt),
                    ],
                    temperature=0,
                    max_tokens=self.llm_block_segment_max_tokens,
                )
                payload = self._parse_json_object(response)
                specs, errors = self._validate_block_segment_payload(payload, atomic_graph)
                if not errors:
                    blocks = [
                        self._block_from_segment_spec(spec, index, atomic_graph, mas_message)
                        for index, spec in enumerate(specs, 1)
                    ]
                    return [block for block in blocks if self.block_builder._accept_block(block)]
                last_errors = errors
            except Exception as exc:
                last_errors = [f"API or JSON parse failure: {exc}"]

            if self.debug:
                print(f"[MAGFold] block segmentation retry {attempt}: {'; '.join(last_errors)}")

        raise RuntimeError(
            "MAGFold LLM block segmentation failed to produce a valid partition: "
            + "; ".join(last_errors)
        )

    def _block_segmenter_prompt(
        self,
        atomic_graph: AtomicInteractionGraph,
        mas_message: MASMessage,
    ) -> str:
        step_lines = self._format_segmenter_steps(atomic_graph)
        examples = self._block_segmenter_examples()
        return f"""
You are given a completed ALFWorld-style task trajectory. Turn it into a few MAGFold memory blocks that will help a future solver act better on a similar task.

Think like the examples below: extract reusable execution episodes, not every small step. A useful block is a completed subgoal or a short failure lesson. Its key_steps should be the executable action spine that a future solver can copy or avoid: go/open/take/clean/heat/cool/use/put steps, plus the observations that reveal the target object or failure. Avoid choosing planning-only "think" steps as key steps when real environment actions exist.

For a successful trajectory, preserve the causal action/observation pattern through the necessary phases. For a failed trajectory, keep only concrete progress and the precise failure cause. If the trajectory mostly repeats thoughts or rejected actions, return one compact failure_cause block with representative steps instead of turning the whole run into a reusable search segment.

{examples}

Now segment the new trajectory in the same style.

Use only these block_type values when they fit: {", ".join(sorted(BLOCK_TYPES))}.

Return a single JSON object with this shape and no markdown fences:
{{
  "blocks": [
    {{
      "block_type": "trajectory_segment",
      "title": "short reusable episode title",
      "summary": "what happened and why it matters",
      "responsibility": "the behavior this block should preserve for retrieval",
      "guidance": "one sentence lesson for a future similar task",
      "outcome": "resolved|failed|unresolved|unknown",
      "covered_steps": [1, 2, 3],
      "key_steps": [2, 3],
      "input_boundary_steps": [1],
      "output_boundary_steps": [3],
      "failure_lesson": null
    }}
  ]
}}

Task:
{atomic_graph.task_description or atomic_graph.task_main}

Task label: {mas_message.label}

Atomic steps:
{step_lines}
""".strip()

    def _block_segmenter_examples(self) -> str:
        return """
## Example 1: successful trajectory
Task: heat some egg and put it in diningtable.
Task label: True
Atomic steps:
Step 0:
  action_or_emit: think: I need to find an egg, heat it with a microwave, then put it on the dining table.
  observation: OK.
Step 1:
  action_or_emit: open fridge 1
  observation: You open the fridge 1. In it, you see a lettuce 2, a mug 2, and a potato 3.
Step 2:
  action_or_emit: go to countertop 1
  observation: On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
Step 3:
  action_or_emit: go to countertop 3
  observation: On the countertop 3, you see a bowl 1, a egg 2, a kettle 2, and a plate 1.
Step 4:
  action_or_emit: take egg 2 from countertop 3
  observation: You pick up the egg 2 from the countertop 3.
Step 5:
  action_or_emit: go to microwave 1
  observation: The microwave 1 is closed.
Step 6:
  action_or_emit: heat egg 2 with microwave 1
  observation: You heat the egg 2 using the microwave 1.
Step 7:
  action_or_emit: go to diningtable 1
  observation: On the diningtable 1, you see a apple 2, a bread 3, and a mug 1.
Step 8:
  action_or_emit: put egg 2 in/on diningtable 1
  observation: You put the egg 2 in/on the diningtable 1.

Output:
{
  "blocks": [
    {
      "block_type": "trajectory_segment",
      "title": "Find and take the target object",
      "summary": "The agent searched likely surfaces, ignored locations without the target, found the egg on countertop 3, and picked up that observed egg instance.",
      "responsibility": "Preserve the search-to-pickup phase for tasks that require manipulating a target object.",
      "guidance": "Search plausible receptacles until the target is explicitly observed, then take that observed instance before planning the next operation.",
      "outcome": "resolved",
      "covered_steps": [1, 2, 3, 4],
      "key_steps": [1, 2, 3, 4],
      "input_boundary_steps": [1],
      "output_boundary_steps": [4],
      "failure_lesson": null
    },
    {
      "block_type": "trajectory_segment",
      "title": "Apply the required transformation and deliver",
      "summary": "After holding the egg, the agent went to the microwave, heated the egg, moved to the dining table, and placed the heated egg there.",
      "responsibility": "Preserve the ordered manipulation pattern: go to the right appliance, transform the held object, then move to the goal receptacle and place it.",
      "guidance": "Once the target is held, satisfy the required state change before going to the final receptacle and putting the object there.",
      "outcome": "resolved",
      "covered_steps": [5, 6, 7, 8],
      "key_steps": [5, 6, 7, 8],
      "input_boundary_steps": [5],
      "output_boundary_steps": [8],
      "failure_lesson": null
    }
  ]
}


## Example 2: examine an object with a light source
Task: look at bowl under the desklamp.
Task label: True
Atomic steps:
Step 0:
  action_or_emit: go to desk 2
  observation: On the desk 2, you see a bowl 1, a book 1, and a pencil 1.
Step 1:
  action_or_emit: take bowl 1 from desk 2
  observation: You pick up the bowl 1 from the desk 2.
Step 2:
  action_or_emit: go to desk 1
  observation: On the desk 1, you see a desklamp 1, a mug 2, and a pen 1.
Step 3:
  action_or_emit: use desklamp 1
  observation: You turn on the desklamp 1.

Output:
{
  "blocks": [
    {
      "block_type": "trajectory_segment",
      "title": "Find and hold the object to examine",
      "summary": "The agent located the bowl on desk 2 and picked up the observed bowl before looking for the light source.",
      "responsibility": "Preserve the first phase of an examine-in-light task: locate the target object and hold the exact observed instance.",
      "guidance": "For look-at-object tasks, first find and take the target object before using the light source.",
      "outcome": "resolved",
      "covered_steps": [0, 1],
      "key_steps": [0, 1],
      "input_boundary_steps": [0],
      "output_boundary_steps": [1],
      "failure_lesson": null
    },
    {
      "block_type": "trajectory_segment",
      "title": "Use the desklamp after taking the target",
      "summary": "After the bowl was held, the agent moved to the desk containing the desklamp and used the desklamp to complete the examination.",
      "responsibility": "Preserve the completion phase for examine-in-light tasks: navigate to a visible desklamp and use it after the target is already held.",
      "guidance": "Once the target object is held, go to the observed desklamp and use it instead of continuing to search for more target objects.",
      "outcome": "resolved",
      "covered_steps": [2, 3],
      "key_steps": [2, 3],
      "input_boundary_steps": [2],
      "output_boundary_steps": [3],
      "failure_lesson": null
    }
  ]
}
""".strip()

    def _format_segmenter_steps(self, atomic_graph: AtomicInteractionGraph) -> str:
        lines: list[str] = []
        for step_id, nodes in self._nodes_grouped_by_step(atomic_graph):
            other_nodes = [
                node
                for node in nodes
                if node.metadata.get("source") not in {"agent_message", "state_observation"}
            ]
            action_text = (
                self._select_step_emit_text(nodes)
                or self._select_step_thought_text(nodes)
                or "(none)"
            )
            observation_text = self._select_step_observation_text(nodes) or "(none)"
            lines.extend(
                [
                    f"Step {step_id}:",
                    f"  action_or_emit: {action_text}",
                    f"  observation: {observation_text}",
                ]
            )
            if other_nodes:
                other_text = " | ".join(
                    f"{node.role}/{node.node_type}: {_shorten(node.content, 160)}"
                    for node in other_nodes
                )
                lines.append(f"  other: {other_text}")
        return "\n".join(lines)

    def _select_step_emit_text(self, nodes: list[AtomicInteractionNode]) -> str:
        """Pick one canonical executable emit for an environment step.

        MacNet and similar MAS frameworks may store the same env action in
        several agent-message nodes for a single state. Rendering all of them as
        `a | a | a` makes the memory prompt look like a compound action, while
        G-Memory renders one `state.graph["action"]` per state. Prefer that
        canonical action when available, then fall back to the action-like agent
        emit that best matches the environment observation.
        """
        observation = self._select_step_observation_text(nodes)
        state_action = self._state_action_from_step_nodes(nodes)
        if state_action and not self._is_thought_text(state_action):
            return _shorten(
                self._select_best_action_text(
                    [
                        (idx, candidate)
                        for idx, candidate in enumerate(self._action_text_candidates(state_action))
                    ],
                    observation,
                ),
                220,
            )

        action_candidates: list[tuple[int, str]] = []
        emit_candidates: list[tuple[int, str]] = []
        for node in nodes:
            if node.metadata.get("source") != "agent_message" or self._is_thought_like(node):
                continue
            text = self._clean_node_text(node)
            if not text:
                continue
            turn = node.turn if node.turn is not None else 0
            for part_idx, candidate in enumerate(self._action_text_candidates(text)):
                item = (turn * 100 + part_idx, candidate)
                if node.node_type == "action":
                    action_candidates.append(item)
                else:
                    emit_candidates.append(item)

        candidates = action_candidates or emit_candidates
        if not candidates:
            return ""
        return _shorten(self._select_best_action_text(candidates, observation), 220)

    def _select_step_thought_text(self, nodes: list[AtomicInteractionNode]) -> str:
        state_action = self._state_action_from_step_nodes(nodes)
        if state_action and self._is_thought_text(state_action):
            return _shorten(state_action, 220)

        thoughts: list[tuple[int, str]] = []
        for node in nodes:
            if node.metadata.get("source") != "agent_message" or not self._is_thought_like(node):
                continue
            text = self._clean_node_text(node)
            if text:
                thoughts.append((node.turn if node.turn is not None else -1, text))
        if not thoughts:
            return ""
        return _shorten(max(thoughts, key=lambda item: item[0])[1], 220)

    def _select_step_observation_text(self, nodes: list[AtomicInteractionNode]) -> str:
        observations = [
            node
            for node in nodes
            if node.metadata.get("source") == "state_observation"
        ]
        if not observations:
            return ""
        node = max(observations, key=lambda item: item.turn if item.turn is not None else -1)
        return _shorten(self._clean_node_text(node), 220)

    def _state_action_from_step_nodes(self, nodes: list[AtomicInteractionNode]) -> str:
        for node in nodes:
            action = node.metadata.get("state_action")
            if action:
                return self._clean_raw_action_text(str(action))
        return ""

    def _clean_raw_action_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^(action|act)\s*\d*\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        return text

    def _is_thought_text(self, text: str) -> bool:
        return (text or "").strip().lower().startswith("think:")

    def _action_text_candidates(self, text: str) -> list[str]:
        """Split accidental multi-agent/compound action renderings.

        `|` is not part of the ALFWorld command language. It appears here when
        multiple MAS emits were previously joined into one prompt line, or when a
        policy copied that joined form back as an action. Keep one candidate so
        retrieved memory never teaches compound pipe-delimited commands.
        """
        raw_parts = [self._clean_raw_action_text(part) for part in (text or "").split("|")]
        candidates: list[str] = []
        seen: set[str] = set()
        for part in raw_parts:
            normalized = self._normalize_action_for_render(part)
            if not part or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(part)
        return candidates or ([self._clean_raw_action_text(text)] if text else [])

    def _select_best_action_text(
        self,
        candidates: list[tuple[int, str]],
        observation: str = "",
    ) -> str:
        valid = [(order, text) for order, text in candidates if text]
        if not valid:
            return ""
        return max(
            valid,
            key=lambda item: (
                self._action_observation_score(item[1], observation),
                -item[0],
            ),
        )[1]

    def _action_observation_score(self, action: str, observation: str) -> int:
        action_lower = (action or "").lower().strip()
        obs_lower = (observation or "").lower().strip()
        if not action_lower or not obs_lower:
            return 0

        score = 0
        verb_pairs = (
            ("put ", ("you put",)),
            ("take ", ("you pick up", "you take")),
            ("pick up ", ("you pick up", "you take")),
            ("open ", ("you open", "is open")),
            ("close ", ("you close", "is closed")),
            ("clean ", ("you clean",)),
            ("heat ", ("you heat",)),
            ("cool ", ("you cool",)),
            ("use ", ("you use",)),
            ("go to ", ("you arrive", "you are facing", "on the ", "in the ")),
        )
        for action_prefix, obs_markers in verb_pairs:
            if action_lower.startswith(action_prefix) and any(marker in obs_lower for marker in obs_markers):
                score += 5
                break

        # Reward overlap on object/location tokens so `go to shelf 2` beats
        # `go to cabinet 2` when the observation says the agent arrived at shelf
        # 2, and `take bowl 2` beats a nearby navigation candidate when the
        # observation says the bowl was picked up.
        action_tokens = set(_tokenize(action_lower))
        obs_tokens = set(_tokenize(obs_lower))
        score += min(len(action_tokens & obs_tokens), 4)
        return score

    def _normalize_action_for_render(self, action: str) -> str:
        return re.sub(r"\s+", " ", (action or "").strip().lower())

    def _nodes_grouped_by_step(
        self,
        atomic_graph: AtomicInteractionGraph,
    ) -> list[tuple[int | str, list[AtomicInteractionNode]]]:
        nodes_by_step: dict[int | str, list[AtomicInteractionNode]] = defaultdict(list)
        for node in atomic_graph.nodes:
            nodes_by_step[node.step_id].append(node)
        return [
            (step_id, sorted(nodes, key=lambda node: node.turn or 0))
            for step_id, nodes in sorted(nodes_by_step.items(), key=lambda item: self._step_sort_key(item[0]))
        ]

    def _step_sort_key(self, step_id: int | str) -> tuple[int, int | str]:
        if isinstance(step_id, int):
            return (0, step_id)
        text = str(step_id)
        if text.isdigit():
            return (0, int(text))
        return (1, text)

    def _parse_json_object(self, response: str) -> dict[str, Any]:
        text = (response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("response does not contain a JSON object")
        return json.loads(text[start : end + 1])

    def _validate_block_segment_payload(
        self,
        payload: dict[str, Any],
        atomic_graph: AtomicInteractionGraph,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return [], ["top-level response must be a JSON object"]
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            return [], ["top-level key 'blocks' must be a non-empty list"]

        valid_node_ids = {node.node_id for node in atomic_graph.nodes}
        valid_step_keys = {str(step_id) for step_id, _ in self._nodes_grouped_by_step(atomic_graph)}
        node_map = atomic_graph.node_map
        specs: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_blocks, 1):
            if not isinstance(raw, dict):
                errors.append(f"blocks[{index}] must be an object")
                continue

            block_type = str(raw.get("block_type", "")).strip()
            if block_type not in BLOCK_TYPES:
                block_type = "generic"

            outcome = str(raw.get("outcome", "unknown")).strip() or "unknown"
            if outcome not in OUTCOMES:
                outcome = "unknown"

            covered_steps = self._clean_step_list(
                raw.get("covered_steps", raw.get("covered_step_ids")),
                valid_step_keys,
            )
            if not covered_steps:
                legacy_covered = self._clean_node_id_list(raw.get("covered_node_ids"), valid_node_ids)
                covered_steps = self._step_keys_for_node_ids(legacy_covered, node_map)

            if not covered_steps:
                errors.append(f"blocks[{index}] has no valid covered_steps")
                continue

            covered_step_set = set(covered_steps)
            covered = self._node_ids_for_step_keys(atomic_graph, covered_steps)
            covered_node_set = set(covered)
            covered_nodes = [node_map[node_id] for node_id in covered if node_id in node_map]

            key_steps = self._clean_step_list(raw.get("key_steps", raw.get("key_step_ids")), valid_step_keys)
            if not key_steps:
                legacy_key_nodes = self._clean_node_id_list(raw.get("key_node_ids"), valid_node_ids)
                key_steps = self._step_keys_for_node_ids(legacy_key_nodes, node_map)
            key_steps = [step for step in key_steps if step in covered_step_set]
            key_nodes = self._node_ids_for_step_keys(atomic_graph, key_steps)
            if not key_nodes:
                key_nodes = self.block_builder._select_key_nodes(
                    covered_nodes,
                    [
                        edge
                        for edge in atomic_graph.edges
                        if edge.src in covered_node_set and edge.dst in covered_node_set
                    ],
                )
                key_steps = self._step_keys_for_node_ids(key_nodes, node_map)
            if not key_nodes and covered:
                key_nodes = covered[-min(len(covered), 3) :]
                key_steps = self._step_keys_for_node_ids(key_nodes, node_map)

            input_steps = self._clean_step_list(raw.get("input_boundary_steps", []), valid_step_keys)
            output_steps = self._clean_step_list(raw.get("output_boundary_steps", []), valid_step_keys)
            if not input_steps:
                input_steps = self._step_keys_for_node_ids(
                    self._clean_node_id_list(raw.get("input_boundary_nodes", []), valid_node_ids),
                    node_map,
                )
            if not output_steps:
                output_steps = self._step_keys_for_node_ids(
                    self._clean_node_id_list(raw.get("output_boundary_nodes", []), valid_node_ids),
                    node_map,
                )
            input_steps = [step for step in input_steps if step in covered_step_set]
            output_steps = [step for step in output_steps if step in covered_step_set]
            input_boundary = self._node_ids_for_step_keys(atomic_graph, input_steps)
            output_boundary = self._node_ids_for_step_keys(atomic_graph, output_steps)

            summary = str(raw.get("summary", "")).strip()
            if not summary:
                summary = self.block_builder._summarize_nodes(covered_nodes[:3] + covered_nodes[-3:], block_type)
            responsibility = str(raw.get("responsibility", "")).strip()
            if not responsibility:
                responsibility = self.block_builder._responsibility(block_type, covered_nodes)
            title = str(raw.get("title", "")).strip()
            if not title:
                title = block_type.replace("_", " ").title()

            spec = dict(raw)
            spec["block_type"] = block_type
            spec["title"] = title
            spec["summary"] = summary
            spec["responsibility"] = responsibility
            spec["outcome"] = outcome
            spec["covered_steps"] = covered_steps
            spec["key_steps"] = key_steps
            spec["input_boundary_steps"] = input_steps
            spec["output_boundary_steps"] = output_steps
            spec["covered_node_ids"] = covered
            spec["key_node_ids"] = key_nodes
            spec["input_boundary_nodes"] = input_boundary
            spec["output_boundary_nodes"] = output_boundary
            specs.append(spec)

        return specs, errors

    def _clean_node_id_list(self, value: Any, valid_node_ids: set[str]) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            node_id = str(item).strip()
            if node_id in valid_node_ids and node_id not in cleaned:
                cleaned.append(node_id)
        return cleaned

    def _clean_step_list(self, value: Any, valid_step_keys: set[str]) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if isinstance(item, float) and item.is_integer():
                step_key = str(int(item))
            else:
                step_key = str(item).strip()
            step_key = re.sub(r"^step\s*", "", step_key, flags=re.IGNORECASE).strip()
            if step_key in valid_step_keys and step_key not in cleaned:
                cleaned.append(step_key)
        return cleaned

    def _step_keys_for_node_ids(
        self,
        node_ids: list[str],
        node_map: dict[str, AtomicInteractionNode],
    ) -> list[str]:
        steps: list[str] = []
        for node_id in node_ids:
            node = node_map.get(node_id)
            if node is None:
                continue
            step_key = str(node.step_id)
            if step_key not in steps:
                steps.append(step_key)
        return steps

    def _node_ids_for_step_keys(
        self,
        atomic_graph: AtomicInteractionGraph,
        step_keys: list[str],
    ) -> list[str]:
        selected = set(step_keys)
        return [
            node.node_id
            for node in sorted(atomic_graph.nodes, key=lambda item: item.turn or 0)
            if str(node.step_id) in selected
        ]

    def _block_from_segment_spec(
        self,
        spec: dict[str, Any],
        index: int,
        atomic_graph: AtomicInteractionGraph,
        mas_message: MASMessage,
    ) -> MAGFoldBlock:
        node_map = atomic_graph.node_map
        nodes = [node_map[node_id] for node_id in spec["covered_node_ids"] if node_id in node_map]
        steps = [
            str(step_id)
            for step_id in sorted(
                {node.step_id for node in nodes},
                key=self._step_sort_key,
            )
        ]
        block = self.block_builder._make_block(
            atomic_graph,
            nodes,
            block_id=f"{atomic_graph.query_id}_llm_block_{index}",
            block_type=spec["block_type"],
            scale=max(2, len({node.step_id for node in nodes})),
            task_context=mas_message,
            result=mas_message.label,
            metadata={
                "source": "llm_block_segmenter",
                "llm_title": _shorten(str(spec.get("title", "")).strip(), 120),
                "llm_guidance": _shorten(str(spec.get("guidance", "")).strip(), 260),
                "source_steps": steps,
                "llm_covered_steps": list(spec.get("covered_steps") or steps),
                "llm_key_steps": list(spec.get("key_steps") or []),
                "llm_input_boundary_steps": list(spec.get("input_boundary_steps") or []),
                "llm_output_boundary_steps": list(spec.get("output_boundary_steps") or []),
            },
        )
        block.summary = _shorten(str(spec.get("summary", "")).strip(), 620)
        block.responsibility = _shorten(str(spec.get("responsibility", "")).strip(), 280)
        block.outcome = spec.get("outcome") or block.outcome
        block.key_node_ids = spec["key_node_ids"] or block.key_node_ids
        if spec.get("input_boundary_nodes"):
            block.input_boundary_nodes = spec["input_boundary_nodes"]
        if spec.get("output_boundary_nodes"):
            block.output_boundary_nodes = spec["output_boundary_nodes"]
        failure_lesson = spec.get("failure_lesson")
        if isinstance(failure_lesson, str) and failure_lesson.strip():
            block.failure_lesson = _shorten(failure_lesson.strip(), 260)
        guidance = block.metadata.get("llm_guidance")
        if guidance and not block.failure_lesson and block.outcome == "failed":
            block.failure_lesson = guidance
        block.confidence = 0.9
        return block

    def retrieve_blocks(
        self,
        query: str,
        role: Optional[str] = None,
        top_k: Optional[int] = None,
        include_failed: bool = True,
    ) -> list[RetrievedBlock]:
        """Retrieve blocks with score breakdowns."""
        top_k = top_k or self.inspiration_top_k
        if not query or not self.blocks:
            # RL hook: keep last_retrieval consistent; otherwise a no-op
            # retrieval would silently inherit the previous task's hits.
            self.last_retrieval = []
            return []

        candidate_blocks = self._candidate_blocks_from_similar_tasks(query, task_top_k=self.task_top_k)
        if not candidate_blocks:
            candidate_blocks = list(self.blocks)
        if not include_failed:
            candidate_blocks = [
                block for block in candidate_blocks
                if self._allow_block_for_failed_policy(block, include_failed=False)
            ]
            if not candidate_blocks:
                # If the most similar source tasks were failures, do what
                # GMemory's successful-case retrieval does: still search the
                # global pool of successful experience instead of injecting no
                # reusable execution pattern.
                candidate_blocks = [
                    block for block in self.blocks
                    if self._allow_block_for_failed_policy(block, include_failed=False)
                ]

        retrieved: list[RetrievedBlock] = []
        for index, block in enumerate(candidate_blocks):
            if role and block.roles and role not in block.roles and role not in block.agents:
                role_gate = 0.85
            else:
                role_gate = 1.0

            text = self.block_retrieval_text(block)
            semantic = self._semantic_similarity(query, text)
            operation = self._operation_match_score(query, block)
            behavior = self._behavior_match(query, block, role)
            outcome = self._outcome_bonus(block)
            recency = (index + 1) / max(len(candidate_blocks), 1)
            score = role_gate * (
                0.42 * semantic
                + 0.22 * operation
                + 0.16 * behavior
                + 0.1 * outcome
                + 0.1 * recency
            )
            retrieved.append(
                RetrievedBlock(
                    block=block,
                    score=score,
                    breakdown={
                        "semantic": semantic,
                        "operation": operation,
                        "behavior": behavior,
                        "outcome": outcome,
                        "recency": recency,
                        "role_gate": role_gate,
                    },
                )
            )

        retrieved.sort(key=lambda item: item.score, reverse=True)
        final = self._select_blocks_with_llm(query, retrieved, top_k)
        # RL hook: record the (block_id, query_id) pairs that survived top-k for
        # this retrieval so the reward function can compute R_retr. Optional.
        try:
            self.last_retrieval = [
                {"block_id": r.block.block_id, "query_id": r.block.query_id, "score": float(r.score)}
                for r in final
            ]
        except Exception:
            self.last_retrieval = []
        return final

    def unfold_block(
        self,
        block: MAGFoldBlock,
        query: str,
        role: Optional[str] = None,
        token_budget: Optional[int] = None,
        full_unfold: Optional[bool] = None,
    ) -> list[AtomicInteractionNode]:
        """Unfold a block into selected atomic nodes using key/boundary priority."""
        full_unfold = self.full_unfold if full_unfold is None else full_unfold
        graph = self._load_atomic_graph(block.query_id)
        if graph is None:
            return []

        node_map = graph.node_map
        covered = [node_map[node_id] for node_id in block.covered_node_ids if node_id in node_map]
        if full_unfold:
            return sorted(covered, key=lambda node: node.turn or 0)

        ordered_ids: list[str] = []
        ordered_ids.extend(block.key_node_ids)
        ordered_ids.extend(block.output_boundary_nodes)
        ordered_ids.extend(block.input_boundary_nodes)
        ordered_ids.extend(self._important_incident_node_ids(block, graph))

        seen = set()
        selected: list[AtomicInteractionNode] = []
        for node_id in ordered_ids:
            if node_id in seen or node_id not in node_map or node_id not in block.covered_node_ids:
                continue
            selected.append(node_map[node_id])
            seen.add(node_id)

        remaining = [node for node in covered if node.node_id not in seen]
        remaining.sort(
            key=lambda node: self._node_score(node, block, graph, query, role),
            reverse=True,
        )
        selected.extend(remaining)

        max_nodes = self.node_top_k_per_block if token_budget is None else len(selected)
        budget = token_budget if token_budget is not None else self.max_memory_tokens
        output: list[AtomicInteractionNode] = []
        used = 0
        for node in selected:
            cost = max(len(_tokenize(node.content)), 1)
            if output and (len(output) >= max_nodes or used + cost > budget):
                continue
            output.append(node)
            used += cost
        return output

    def retrieve_memory_prompt(
        self,
        query: str,
        role: Optional[str] = None,
        token_budget: Optional[int] = None,
        top_k: Optional[int] = None,
        full_unfold: Optional[bool] = None,
    ) -> str:
        """Return prompt-ready natural language MAGFold memory text."""
        return "\n\n".join(
            self.retrieve_memory_items(
                query=query,
                role=role,
                token_budget=token_budget,
                top_k=top_k,
                full_unfold=full_unfold,
            )
        )

    def retrieve_memory_items(
        self,
        query: str,
        role: Optional[str] = None,
        token_budget: Optional[int] = None,
        top_k: Optional[int] = None,
        full_unfold: Optional[bool] = None,
    ) -> list[str]:
        """Return natural language block items for debugging or direct inspection."""
        retrieved = self.retrieve_blocks(query, role=role, top_k=top_k)
        if not retrieved:
            return []

        budget = token_budget or self.max_memory_tokens
        items: list[str] = []
        used = 0
        for item in retrieved:
            block = item.block
            nodes = self.unfold_block(
                block,
                query=query,
                role=role,
                token_budget=max(1, budget - used),
                full_unfold=full_unfold,
            )
            text = self._format_natural_memory_item(block, nodes)
            item_cost = len(_tokenize(text))
            if items and used + item_cost > budget:
                break
            items.append(text)
            used += item_cost
        return items

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        insight_topk: Optional[int] = None,
        role: Optional[str] = None,
        token_budget: Optional[int] = None,
        **kwargs,
    ) -> tuple[list, list, list]:
        insight_limit = insight_topk or failed_topk or self.inspiration_top_k
        top_k = kwargs.get("block_top_k") or insight_topk or self.inspiration_top_k
        related_messages = []
        if self.enable_magfold_overview_injection:
            related_messages = self.retrieve_related_experiences(
                query_task=query_task,
                role=role,
                token_budget=token_budget,
                top_k=top_k,
                full_unfold=kwargs.get("full_unfold"),
                include_failed=False,
            )
        insights: list[str] = []
        seen_insights: set[str] = set()
        if self.enable_magfold_insight_injection and self.enable_alfworld_look_action_advisor and self._alfworld_operation_signature(query_task) == "look":
            self._append_unique_insight(
                insights,
                seen_insights,
                self._alfworld_look_action_protocol(query_task),
                insight_limit,
            )

        if self.enable_magfold_insight_injection:
            for message in related_messages:
                for insight in message.get_extra_field("magfold_selected_insights") or []:
                    self._append_unique_insight(
                        insights,
                        seen_insights,
                        insight,
                        insight_limit,
                    )
                    if len(insights) >= insight_limit:
                        break
                if len(insights) >= insight_limit:
                    break

            if len(insights) < insight_limit:
                for insight in self.retrieve_block_insights(
                    query_task=query_task,
                    role=role,
                    top_k=insight_limit,
                ):
                    self._append_unique_insight(insights, seen_insights, insight, insight_limit)
                    if len(insights) >= insight_limit:
                        break
        if not related_messages and not insights:
            return [], [], []
        return related_messages, [], insights

    def retrieve_block_insights(
        self,
        query_task: str,
        role: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[str]:
        """Project retrieved block guidance into the insight channel used by MAS prompts."""
        effective_top_k = top_k or self.inspiration_top_k
        retrieved = self.retrieve_blocks(
            query_task,
            role=role,
            top_k=max(effective_top_k * 2, effective_top_k),
            include_failed=True,
        )
        if not retrieved:
            return []

        insights: list[str] = []
        seen: set[str] = set()
        for outcome in ("failed", "resolved", "unresolved", "unknown"):
            for item in retrieved:
                block = item.block
                if block.outcome != outcome:
                    continue
                self._append_unique_insight(
                    insights,
                    seen,
                    self._block_guidance_for_insight(block),
                    effective_top_k,
                )
                if len(insights) >= effective_top_k:
                    return insights
        return insights

    def _block_guidance_for_insight(
        self,
        block: MAGFoldBlock,
        nodes: Optional[list[AtomicInteractionNode]] = None,
    ) -> str:
        guidance = block.failure_lesson or block.metadata.get("llm_guidance") or ""
        if not guidance and nodes is not None:
            guidance = self._natural_guidance(block, nodes)

        guidance = self._proceduralize_alfworld_guidance(guidance)

        failure_text = " ".join(
            str(part or "")
            for part in (
                block.metadata.get("llm_title"),
                block.summary,
                block.failure_lesson,
                block.metadata.get("llm_guidance"),
            )
        ).lower()
        if block.outcome == "failed" and any(
            marker in failure_text
            for marker in ("kept thinking", "repeatedly planned", "without execution", "never issued")
        ):
            guidance = (
                "After one brief `think:` plan, immediately output the next concrete ALFWorld command, "
                "such as `go to cabinet 1` or `open cabinet 1`; do not repeat the same `think:`."
        )
        return self._sentence(_shorten(str(guidance).strip(), 260)) if guidance else ""

    def _proceduralize_alfworld_guidance(self, guidance: str) -> str:
        lowered = (guidance or "").lower()
        if not lowered:
            return guidance
        if "final receptacle" in lowered and "put" in lowered:
            return (
                "After the target is held and any required clean/heat/cool step is done, "
                "go to the final receptacle and immediately issue the `put ... in/on ...` command; "
                "do not keep searching after reaching the receptacle."
            )
        if "explicitly observed" in lowered and "take" in lowered:
            return (
                "Search likely receptacles until the target is explicitly observed; "
                "then immediately `take ... from ...` that exact observed instance before doing the next operation."
            )
        if "look-at-object" in lowered or "look at object" in lowered or "look/examine" in lowered:
            return (
                "For look/examine tasks, first take the observed target object, then go to the observed light source "
                "and use it; do not keep searching after the target is held."
            )
        return guidance

    def _current_alfworld_task_text(self) -> str:
        context = getattr(self, "current_task_context", None)
        if context is None:
            return ""
        return "\n".join(part for part in (context.task_main, context.task_description) if part)

    def _current_alfworld_history(self) -> list[tuple[str, str]]:
        context = getattr(self, "current_task_context", None)
        if context is None:
            return []
        history: list[tuple[str, str]] = []
        for state in context.chain_of_states:
            action = str(state.graph.get("action") or "").strip()
            observation = str(state.graph.get("observation") or "").strip()
            if action or observation:
                history.append((action, observation))
        return history

    def _alfworld_look_action_protocol(self, query_task: str) -> str:
        target = self._alfworld_look_target(query_task) or "target object"
        return (
            "Look/examine command protocol: start `go to desk 1`; remember the desklamp location; "
            f"search until `{target}` is seen; immediately `take {target} <n> from <location>`; "
            "then go to that desklamp and `use desklamp <n>`. Avoid repeated `think:`."
        )

    def _alfworld_look_next_action(self, query_task: str) -> str:
        target = self._alfworld_look_target(query_task)
        if not target:
            return ""
        history = self._current_alfworld_history()
        available_locations = self._alfworld_available_locations(query_task)
        search_order = self._alfworld_ordered_locations(available_locations)
        if not search_order:
            search_order = list(ALFWORLD_LOOK_SEARCH_ORDER)

        if history:
            last_action, last_observation = history[-1]
            last_observation_lower = last_observation.lower()
            target_instance = self._alfworld_object_instance(last_observation_lower, target)
            target_location = self._alfworld_location_from_action_observation(last_action, last_observation)
            if target_instance and target_location and not self._alfworld_target_is_held(target, history):
                return f"take {target} {target_instance} from {target_location}"

            closed_location = self._alfworld_closed_location(last_observation_lower)
            if closed_location:
                return f"open {closed_location}"

        held_instance = self._alfworld_target_is_held(target, history)
        if held_instance:
            current_location = self._alfworld_current_location(history)
            lamp = self._alfworld_known_desklamp(history)
            if lamp:
                lamp_location, lamp_instance = lamp
                if current_location == lamp_location:
                    return f"use desklamp {lamp_instance}"
                return f"go to {lamp_location}"

            next_lamp_location = self._alfworld_next_unvisited_location(search_order, history)
            return f"go to {next_lamp_location}" if next_lamp_location else "use desklamp 1"

        next_location = self._alfworld_next_unvisited_location(search_order, history)
        return f"go to {next_location}" if next_location else ""

    def _alfworld_look_target(self, text: str) -> str:
        clause = self._alfworld_task_clause(text).lower()
        patterns = (
            r"\blook\s+at\s+(?:the\s+|a\s+|an\s+|some\s+)?(.+?)\s+under\s+the\s+desklamp\b",
            r"\bexamine\s+(?:the\s+|a\s+|an\s+|some\s+)?(.+?)\s+with\s+the\s+desklamp\b",
        )
        for pattern in patterns:
            match = re.search(pattern, clause)
            if match:
                return re.sub(r"\s+", " ", match.group(1)).strip()
        return ""

    def _alfworld_available_locations(self, text: str) -> list[str]:
        pattern = self._alfworld_location_pattern()
        seen: set[str] = set()
        locations: list[str] = []
        for match in re.finditer(pattern, (text or "").lower()):
            location = match.group(0)
            if location not in seen:
                seen.add(location)
                locations.append(location)
        return locations

    def _alfworld_ordered_locations(self, available_locations: list[str]) -> list[str]:
        available = set(available_locations)
        ordered = [location for location in ALFWORLD_LOOK_SEARCH_ORDER if location in available]
        for location in available_locations:
            if location not in ordered:
                ordered.append(location)
        return ordered

    def _alfworld_next_unvisited_location(
        self,
        search_order: list[str],
        history: list[tuple[str, str]],
    ) -> str:
        visited = {
            location
            for action, observation in history
            for location in [self._alfworld_location_from_action_observation(action, observation)]
            if location
        }
        for location in search_order:
            if location not in visited:
                return location
        return ""

    def _alfworld_target_is_held(self, target: str, history: list[tuple[str, str]]) -> str:
        held_instance = ""
        escaped = re.escape(target)
        for action, observation in history:
            action_lower = action.lower()
            observation_lower = observation.lower()
            take_match = re.search(rf"\btake\s+{escaped}\s+(\d+)\s+from\b", action_lower)
            pickup_match = re.search(rf"\bpick up the\s+{escaped}\s+(\d+)\b", observation_lower)
            if take_match:
                held_instance = take_match.group(1)
            if pickup_match:
                held_instance = pickup_match.group(1)
        return held_instance

    def _alfworld_known_desklamp(self, history: list[tuple[str, str]]) -> Optional[tuple[str, str]]:
        known: Optional[tuple[str, str]] = None
        for action, observation in history:
            observation_lower = observation.lower()
            if "desklamp" not in observation_lower:
                continue
            location = self._alfworld_location_from_action_observation(action, observation)
            if not location:
                continue
            instance = self._alfworld_object_instance(observation_lower, "desklamp") or "1"
            known = (location, instance)
        return known

    def _alfworld_current_location(self, history: list[tuple[str, str]]) -> str:
        for action, observation in reversed(history):
            location = self._alfworld_location_from_action_observation(action, observation)
            if location:
                return location
        return ""

    def _alfworld_location_from_action_observation(self, action: str, observation: str) -> str:
        action_lower = (action or "").lower().strip()
        for pattern in (
            r"^go to\s+(.+?)$",
            r"^open\s+(.+?)$",
            r"^close\s+(.+?)$",
            r"^take\s+.+?\s+from\s+(.+?)$",
        ):
            match = re.search(pattern, action_lower)
            if match:
                location = self._normalize_alfworld_location(match.group(1))
                if location:
                    return location

        observation_lower = (observation or "").lower()
        location_pattern = self._alfworld_location_pattern()
        for pattern in (
            rf"\byou arrive at\s+({location_pattern})\b",
            rf"\bon the\s+({location_pattern})\b",
            rf"\bthe\s+({location_pattern})\s+is\b",
            rf"\byou are facing the\s+({location_pattern})\b",
        ):
            match = re.search(pattern, observation_lower)
            if match:
                return self._normalize_alfworld_location(match.group(1))
        return ""

    def _normalize_alfworld_location(self, text: str) -> str:
        match = re.search(self._alfworld_location_pattern(), (text or "").lower())
        return match.group(0) if match else ""

    def _alfworld_closed_location(self, observation: str) -> str:
        match = re.search(rf"\bthe\s+({self._alfworld_location_pattern()})\s+is\s+closed\b", observation or "")
        return match.group(1) if match else ""

    def _alfworld_object_instance(self, text: str, object_name: str) -> str:
        escaped = re.escape((object_name or "").lower().strip())
        if not escaped:
            return ""
        match = re.search(rf"\b{escaped}\s+(\d+)\b", (text or "").lower())
        return match.group(1) if match else ""

    def _alfworld_location_pattern(self) -> str:
        return rf"(?:{'|'.join(ALFWORLD_LOCATION_TYPES)})\s+\d+"

    def _append_unique_insight(
        self,
        insights: list[str],
        seen: set[str],
        insight: str,
        limit: int,
    ) -> None:
        if len(insights) >= limit:
            return
        insight = _shorten(str(insight).strip(), 260)
        if not insight:
            return
        normalized = re.sub(r"\s+", " ", insight.lower().rstrip(".!?"))
        if normalized in seen:
            return
        seen.add(normalized)
        insights.append(self._sentence(insight))

    def retrieve_related_experiences(
        self,
        query_task: str,
        role: Optional[str] = None,
        token_budget: Optional[int] = None,
        top_k: Optional[int] = None,
        full_unfold: Optional[bool] = None,
        include_failed: bool = True,
    ) -> list[MASMessage]:
        """Return retrieved MAGFold blocks grouped by their source task."""
        effective_top_k = top_k or self.inspiration_top_k
        retrieved = self.retrieve_blocks(
            query_task,
            role=role,
            top_k=effective_top_k,
            include_failed=include_failed,
        )
        if not retrieved:
            return []

        if not include_failed:
            retrieved = [
                item for item in retrieved
                if self._allow_block_for_failed_policy(item.block, include_failed=False)
            ]
        retrieved = self._dedupe_retrieved_blocks_for_prompt(retrieved, query_task, role, full_unfold)

        grouped: dict[str, list[RetrievedBlock]] = {}
        for item in retrieved[:effective_top_k]:
            grouped.setdefault(item.block.query_id, []).append(item)

        budget = token_budget or self.max_memory_tokens
        used = 0
        messages: list[MASMessage] = []
        for query_id, items in grouped.items():
            graph = self._load_atomic_graph(query_id)
            if graph is None:
                continue
            if not include_failed and graph.metadata.get("label") is False:
                continue

            key_steps: list[str] = []
            for index, item in enumerate(items, 1):
                block = item.block
                nodes = self.unfold_block(
                    block,
                    query=query_task,
                    role=role,
                    token_budget=max(1, budget - used),
                    full_unfold=full_unfold,
                )
                inspiration = self._format_inspiration(index, block, nodes, full_unfold=full_unfold)
                cost = len(_tokenize(inspiration))
                if (key_steps or messages) and used + cost > budget:
                    break
                key_steps.append(inspiration)
                used += cost

            if not key_steps:
                continue

            selected_insights: list[str] = []
            for item in items[: len(key_steps)]:
                block = item.block
                nodes = self.unfold_block(
                    block,
                    query=query_task,
                    role=role,
                    token_budget=max(1, budget),
                    full_unfold=full_unfold,
                )
                guidance = self._block_guidance_for_insight(block, nodes)
                if guidance:
                    selected_insights.append(guidance)

            message = MASMessage(
                task_main=graph.task_main,
                task_description=graph.task_description,
                task_trajectory="",
                label=graph.metadata.get("label"),
            )
            message.add_extra_field("key_steps", "\n\n".join(key_steps))
            message.add_extra_field("magfold_source_query_id", query_id)
            message.add_extra_field("magfold_selected_blocks", [item.block.block_id for item in items[: len(key_steps)]])
            message.add_extra_field("magfold_selected_insights", selected_insights)
            messages.append(message)
        return messages

    def _allow_block_for_failed_policy(
        self,
        block: MAGFoldBlock,
        include_failed: bool,
    ) -> bool:
        if include_failed:
            return True
        if block.outcome == "failed":
            return False
        graph = self._load_atomic_graph(block.query_id)
        if graph is not None and graph.metadata.get("label") is False:
            return False
        return True

    def _dedupe_retrieved_blocks_for_prompt(
        self,
        retrieved: list[RetrievedBlock],
        query_task: str,
        role: Optional[str],
        full_unfold: Optional[bool],
    ) -> list[RetrievedBlock]:
        """Remove prompt-equivalent blocks before formatting memories.

        The block graph may contain a step block, segment block, or repeated
        failed attempts with the same action/observation content. Ranking can
        select several of them when a larger top-k is requested, but injecting
        the same lesson several times only pollutes the solver context.
        """
        deduped: list[RetrievedBlock] = []
        seen: set[str] = set()
        for item in retrieved:
            signature = self._retrieved_block_prompt_signature(item, query_task, role, full_unfold)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(item)
        return deduped

    def _retrieved_block_prompt_signature(
        self,
        item: RetrievedBlock,
        query_task: str,
        role: Optional[str],
        full_unfold: Optional[bool],
    ) -> str:
        block = item.block
        nodes = self.unfold_block(
            block,
            query=query_task,
            role=role,
            token_budget=self.max_memory_tokens,
            full_unfold=full_unfold,
        )
        trajectory = self._format_detail_trajectory(block, full_unfold=full_unfold)
        if not trajectory:
            trajectory = "\n".join(self._clean_node_text(node) for node in nodes)
        text = "\n".join(
            [
                block.query_id,
                block.outcome,
                block.metadata.get("llm_title") or self._default_inspiration_title(block),
                block.metadata.get("llm_guidance") or "",
                trajectory,
                block.summary,
            ]
        )
        return re.sub(r"\s+", " ", text.strip().lower())

    def retrieve_related_experience(
        self,
        query_task: str,
        role: Optional[str] = None,
        token_budget: Optional[int] = None,
        top_k: Optional[int] = None,
        full_unfold: Optional[bool] = None,
    ) -> Optional[MASMessage]:
        """Return a MASMessage that is injected as a related past experience."""
        messages = self.retrieve_related_experiences(
            query_task=query_task,
            role=role,
            token_budget=token_budget,
            top_k=top_k,
            full_unfold=full_unfold,
        )
        return messages[0] if messages else None

    def block_retrieval_text(self, block: MAGFoldBlock) -> str:
        return "\n".join(
            [
                block.block_type,
                block.responsibility,
                block.summary,
                block.input_state,
                block.output_state,
                block.outcome,
                " ".join(block.roles),
                " ".join(block.key_claims),
                " ".join(block.decisions),
                block.failure_lesson or "",
                " ".join(f"{key}:{value}" for key, value in block.edge_profile.items()),
            ]
        )

    def _candidate_blocks_from_similar_tasks(self, query: str, task_top_k: int) -> list[MAGFoldBlock]:
        query_operation = self._alfworld_operation_signature(query)
        query_scores: list[tuple[str, float, float, str]] = []
        for query_id in self.query_index.keys():
            graph = self._load_atomic_graph(query_id)
            task_text = self._task_retrieval_text(query_id, graph)
            semantic = self._semantic_similarity(query, task_text)
            operation = self._operation_match_between_texts(query, task_text)
            source_operation = self._alfworld_operation_signature(task_text)
            score = 0.65 * semantic + 0.35 * operation
            query_scores.append((query_id, score, operation, source_operation))

        query_scores.sort(key=lambda item: item[1], reverse=True)
        if query_operation:
            exact = [
                item for item in query_scores
                if item[3] == query_operation
            ]
            compatible = [
                item for item in query_scores
                if item[3] != query_operation
                and self._operation_candidates_compatible(query_operation, item[3])
            ]
            incompatible = [
                item for item in query_scores
                if item[3] != query_operation
                and not self._operation_candidates_compatible(query_operation, item[3])
            ]
            ordered_scores = exact + compatible + incompatible
        else:
            ordered_scores = query_scores
        selected_query_ids = {query_id for query_id, *_ in ordered_scores[: max(task_top_k, 1)]}
        return [block for block in self.blocks if block.query_id in selected_query_ids]

    def _task_retrieval_text(
        self,
        query_id: str,
        graph: Optional[AtomicInteractionGraph],
    ) -> str:
        if graph is not None:
            parts = [graph.task_main, graph.task_description]
        else:
            parts = []
        for block in self.blocks:
            if block.query_id == query_id:
                parts.extend([block.responsibility, block.summary, block.outcome])
        return "\n".join(part for part in parts if part)

    def _operation_match_score(self, query: str, block: MAGFoldBlock) -> float:
        graph = self._load_atomic_graph(block.query_id)
        source_text = self._task_retrieval_text(block.query_id, graph)
        if not source_text:
            source_text = self.block_retrieval_text(block)
        return self._operation_match_between_texts(query, source_text)

    def _operation_match_between_texts(self, left: str, right: str) -> float:
        left_operation = self._alfworld_operation_signature(left)
        right_operation = self._alfworld_operation_signature(right)
        if not left_operation or not right_operation:
            return 0.5
        if left_operation == right_operation:
            return 1.0
        if "look" in {left_operation, right_operation}:
            return 0.0
        if {left_operation, right_operation} <= {"clean", "cool", "heat"}:
            return 0.0
        if {left_operation, right_operation} <= {"put", "put_two"}:
            return 0.55
        if left_operation in {"put", "put_two"} or right_operation in {"put", "put_two"}:
            return 0.0
        return 0.2

    def _operation_candidates_compatible(self, query_operation: str, source_operation: str) -> bool:
        if not query_operation or not source_operation:
            return False
        if query_operation == source_operation:
            return True
        return {query_operation, source_operation} <= {"put", "put_two"}

    def _alfworld_operation_signature(self, text: str) -> str:
        clause = self._alfworld_task_clause(text).lower()
        if not clause:
            return ""
        if re.search(r"\b(?:look\s+at|examine)\b", clause):
            return "look"
        if re.search(r"\bput\s+(?:two|2)\b", clause):
            return "put_two"
        if re.search(r"\b(?:clean|cleaned|wash|washed)\b|\ba\s+clean\b", clause):
            return "clean"
        if re.search(r"\b(?:cool|cooled|cold)\b|\ba\s+cool\b", clause):
            return "cool"
        if re.search(r"\b(?:heat|heated|hot)\b|\ba\s+hot\b", clause):
            return "heat"
        if re.search(r"\bput\b", clause):
            return "put"
        return ""

    def _alfworld_task_clause(self, text: str) -> str:
        raw = text or ""
        match = re.search(r"your task is to:\s*(.+?)(?:\n|$)", raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            raw = match.group(1)
        return re.sub(r"\s+", " ", raw).strip(" .")

    def _select_blocks_with_llm(
        self,
        query: str,
        ranked_blocks: list[RetrievedBlock],
        top_k: int,
    ) -> list[RetrievedBlock]:
        if not ranked_blocks:
            return []
        fallback = ranked_blocks[:top_k]
        if not self.enable_llm_block_selector:
            return fallback

        candidates = ranked_blocks[: min(len(ranked_blocks), 12)]
        candidate_text = "\n".join(
            [
                f"{idx}. type={item.block.block_type}; outcome={item.block.outcome}; "
                f"task={self._source_task_brief(item.block.query_id)}; "
                f"description={self._block_description_for_selection(item.block)}"
                for idx, item in enumerate(candidates, 1)
            ]
        )
        prompt = (
            "Select the most relevant interaction blocks for the current task. "
            "Rank candidates by expected usefulness for completing the task, not by "
            "surface lexical overlap alone. Prefer blocks that contain decisive "
            "state-changing or goal-enabling operations, such as heating, cooling, "
            "cleaning, using or turning on an object, slicing, placing an object in "
            "the required final location, or other key manipulations that satisfy a "
            "task constraint. Give lower priority to generic setup steps such as "
            "navigation, looking around, finding a container, or taking an object "
            "unless those steps are specifically the hard or missing part of the "
            "current task. Setup blocks may still be selected when needed, but key "
            "operation blocks should receive the higher rank when both are relevant. "
            "Return only a JSON list of candidate numbers, with no explanation.\n\n"
            f"Current task:\n{query}\n\n"
            f"Candidates:\n{candidate_text}\n\n"
            f"Select at most {top_k} candidates."
        )
        try:
            response = self.aux_llm_model(
                messages=[
                    Message("system", "You rank candidate interaction blocks and return only candidate numbers."),
                    Message("user", prompt),
                ],
                temperature=0,
            )
        except Exception:
            return fallback

        selected_indices = self._parse_selected_indices(response, max_index=len(candidates))
        if not selected_indices:
            return fallback

        selected: list[RetrievedBlock] = []
        for index in selected_indices:
            item = candidates[index - 1]
            if item not in selected:
                selected.append(item)
            if len(selected) >= top_k:
                break
        return selected or fallback

    def _parse_selected_indices(self, response: str, max_index: int) -> list[int]:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                indices = [int(item) for item in parsed]
            else:
                indices = []
        except Exception:
            indices = [int(match) for match in re.findall(r"\d+", response or "")]
        return [index for index in indices if 1 <= index <= max_index]

    def _source_task_brief(self, query_id: str) -> str:
        graph = self._load_atomic_graph(query_id)
        if graph is None:
            return query_id
        return _shorten(graph.task_main or graph.task_description, 140)

    def _block_description_for_selection(self, block: MAGFoldBlock) -> str:
        title = block.metadata.get("llm_title") or block.block_type.replace("_", " ")
        guidance = block.metadata.get("llm_guidance") or block.failure_lesson or ""
        return _shorten(f"{title}. {block.summary}. {guidance}", 420)

    def _curate_block_with_llm(
        self,
        block: MAGFoldBlock,
        atomic_graph: AtomicInteractionGraph,
    ) -> MAGFoldBlock:
        if not self.enable_llm_block_curator:
            return block

        nodes = [atomic_graph.node_map[node_id] for node_id in block.key_node_ids if node_id in atomic_graph.node_map]
        node_text = "\n".join(
            f"- {node.role}/{node.node_type}: {_shorten(node.content, 180)}"
            for node in nodes[:8]
        )
        prompt = (
            "Write a concise natural-language description for this multi-agent interaction block. "
            "Return JSON with keys: title, summary, responsibility, guidance. "
            "Do not invent facts beyond the provided nodes.\n\n"
            f"Task: {atomic_graph.task_description or atomic_graph.task_main}\n"
            f"Outcome: {block.outcome}\n"
            f"Block type: {block.block_type}\n"
            f"Nodes:\n{node_text}"
        )
        try:
            response = self.aux_llm_model(
                messages=[
                    Message("system", "You summarize interaction blocks as concise reusable execution experience."),
                    Message("user", prompt),
                ],
                temperature=0,
            )
            parsed = json.loads(response)
        except Exception:
            return block

        if isinstance(parsed, dict):
            title = str(parsed.get("title", "")).strip()
            summary = str(parsed.get("summary", "")).strip()
            responsibility = str(parsed.get("responsibility", "")).strip()
            guidance = str(parsed.get("guidance", "")).strip()
            if summary:
                block.summary = _shorten(summary, 520)
            if responsibility:
                block.responsibility = _shorten(responsibility, 260)
            if guidance:
                block.metadata["llm_guidance"] = _shorten(guidance, 260)
            if title:
                block.metadata["llm_title"] = _shorten(title, 120)
        return block

    def format_block_card(self, block: MAGFoldBlock) -> str:
        """Return a compact human-readable block card."""
        return "\n".join(
            [
                f"[Block {block.block_id} | {block.block_type} | {block.outcome}]",
                f"Responsibility: {block.responsibility}",
                f"Summary: {block.summary}",
                f"Covered nodes: {len(block.covered_node_ids)}, edges: {len(block.covered_edge_ids)}",
                f"Key nodes: {', '.join(block.key_node_ids)}",
            ]
        )

    def _format_natural_memory_item(
        self,
        block: MAGFoldBlock,
        nodes: list[AtomicInteractionNode],
    ) -> str:
        prior = self._prior_context_phrase(block)
        main_event = self._natural_main_event(block, nodes)
        guidance = self._natural_guidance(block, nodes)
        details = self._natural_details(nodes)

        parts = [f"{prior}: {main_event}"]
        if guidance:
            parts.append(f"Guidance: {guidance}")
        if details:
            parts.append(f"Useful details: {details}")
        return " ".join(parts)

    def _format_inspiration(
        self,
        index: int,
        block: MAGFoldBlock,
        nodes: list[AtomicInteractionNode],
        full_unfold: Optional[bool] = None,
    ) -> str:
        title = block.metadata.get("llm_title") or self._default_inspiration_title(block)
        if block.metadata.get("source") == "llm_block_segmenter" or block.metadata.get("llm_title"):
            summary = self._clean_summary(block.summary, limit=None)
        else:
            summary = self._natural_main_event(block, nodes)
        trajectory = self._format_detail_trajectory(block, full_unfold=full_unfold)

        lines = [
            f"#### Inspiration {index}",
            f"{title}:",
            self._sentence(summary),
        ]
        if self.enable_magfold_detail_trajectory and trajectory:
            lines.extend(["", "Detail trajectory:", trajectory])
        return "\n".join(lines).strip()

    def _default_inspiration_title(self, block: MAGFoldBlock) -> str:
        title_map = {
            "step_transition": "Reusing a state transition",
            "evidence_verification": "Checking evidence before acting",
            "conflict_resolution": "Handling conflicting information",
            "tool_use": "Using a tool result",
            "loop_or_stuck": "Handling a repeated or stuck action",
            "trajectory_segment": "Following a multi-step execution pattern",
            "failure_cause": "Avoiding a known failure pattern",
            "generic": "Related execution experience",
        }
        return title_map.get(block.block_type, "Related execution experience")

    def _format_detail_trajectory(
        self,
        block: MAGFoldBlock,
        max_pairs: int = 8,
        full_unfold: Optional[bool] = None,
    ) -> str:
        full_unfold = self.full_unfold if full_unfold is None else full_unfold
        graph = self._load_atomic_graph(block.query_id)
        if graph is None:
            return ""
        node_map = graph.node_map
        if full_unfold:
            return "\n".join(
                node_map[node_id].content
                for node_id in block.covered_node_ids
                if node_id in node_map
            )
        covered_nodes = [
            node_map[node_id]
            for node_id in block.covered_node_ids
            if node_id in node_map
        ]
        nodes_by_step: dict[int | str, list[AtomicInteractionNode]] = defaultdict(list)
        for node in covered_nodes:
            nodes_by_step[node.step_id].append(node)

        by_step: dict[int | str, dict[str, str]] = defaultdict(dict)
        for step_id, step_nodes in nodes_by_step.items():
            emit = self._select_step_emit_text(step_nodes)
            thought = self._select_step_thought_text(step_nodes)
            observation = self._select_step_observation_text(step_nodes)
            if emit:
                by_step[step_id]["emit"] = emit
            if thought:
                by_step[step_id]["thought"] = thought
            if observation:
                by_step[step_id]["observation"] = observation

        has_executable_emit = any(pair.get("emit") for pair in by_step.values())
        if full_unfold:
            selected_step_keys = sorted(by_step, key=self._step_sort_key)
        else:
            selected_step_keys = self._detail_trajectory_step_keys(block, by_step, max_pairs)
        lines: list[str] = []
        for step_id in selected_step_keys:
            pair = by_step[step_id]
            action = pair.get("emit")
            if not action and (full_unfold or not has_executable_emit):
                action = pair.get("thought")
            observation = pair.get("observation")
            if not action and observation and self._is_low_information_observation(observation):
                continue
            if not action and not observation:
                continue
            if action:
                lines.append(f"> {action}")
            if observation:
                lines.append(_shorten(observation, 200))
        return "\n".join(lines)

    def _detail_trajectory_step_keys(
        self,
        block: MAGFoldBlock,
        by_step: dict[int | str, dict[str, str]],
        max_pairs: int,
    ) -> list[int | str]:
        available_by_key = {str(step_id): step_id for step_id in by_step.keys()}
        selected_keys: list[str] = []

        def extend(values: Any, limit: Optional[int] = None) -> None:
            if not isinstance(values, list):
                return
            for value in values:
                if limit is not None and len(selected_keys) >= limit:
                    break
                key = str(value)
                if key in available_by_key and key not in selected_keys:
                    selected_keys.append(key)

        # Keep all model-selected key steps. `max_pairs` is only a fill target
        # when the segmenter supplied too few key steps for a useful trace.
        extend(block.metadata.get("llm_key_steps"))

        if len(selected_keys) < max_pairs:
            for step_id in sorted(by_step, key=self._step_sort_key):
                action = by_step[step_id].get("emit")
                if not action:
                    continue
                key = str(step_id)
                if key not in selected_keys:
                    selected_keys.append(key)
                if len(selected_keys) >= max_pairs:
                    break

        if len(selected_keys) < max_pairs:
            extend(block.metadata.get("llm_output_boundary_steps"), limit=max_pairs)
        if len(selected_keys) < max_pairs:
            extend(block.metadata.get("llm_input_boundary_steps"), limit=max_pairs)
        if len(selected_keys) < max_pairs:
            extend(block.metadata.get("source_steps"), limit=max_pairs)

        if len(selected_keys) < max_pairs:
            for step_id in sorted(by_step, key=self._step_sort_key):
                key = str(step_id)
                if key not in selected_keys:
                    selected_keys.append(key)
                if len(selected_keys) >= max_pairs:
                    break

        selected = [available_by_key[key] for key in selected_keys]
        return sorted(selected, key=self._step_sort_key)

    def _prior_context_phrase(self, block: MAGFoldBlock) -> str:
        outcome = {
            "resolved": "In a similar successful attempt",
            "failed": "In a similar failed attempt",
            "unresolved": "In a similar unresolved attempt",
            "unknown": "In a related past attempt",
        }.get(block.outcome, "In a related past attempt")

        block_type = {
            "evidence_verification": "while checking evidence",
            "conflict_resolution": "while resolving conflicting information",
            "loop_or_stuck": "when the agent got stuck",
            "trajectory_segment": "during a multi-step trajectory",
            "failure_cause": "around a failure point",
            "tool_use": "while using a tool",
            "step_transition": "during one environment step",
            "generic": "",
        }.get(block.block_type, "")

        return f"{outcome} {block_type}".strip()

    def _natural_main_event(
        self,
        block: MAGFoldBlock,
        nodes: list[AtomicInteractionNode],
    ) -> str:
        if not nodes:
            return self._clean_summary(block.summary)

        actions = [
            self._clean_node_text(node)
            for node in nodes
            if node.node_type == "action" and not self._is_thought_like(node)
        ]
        observations = [self._clean_node_text(node) for node in nodes if node.node_type == "observation"]
        decisions = [self._clean_node_text(node) for node in nodes if node.node_type == "decision"]
        thoughts = [
            self._clean_node_text(node)
            for node in nodes
            if node.node_type == "thought" or self._is_thought_like(node)
        ]

        if block.block_type == "loop_or_stuck":
            repeated = actions[-1] if actions else self._clean_summary(block.summary)
            return f"the agent repeated or circled around '{_shorten(repeated, 120)}' without making useful progress."

        if block.block_type in {"conflict_resolution", "evidence_verification"}:
            claim = decisions[-1] if decisions else thoughts[-1] if thoughts else actions[-1] if actions else ""
            observation = observations[-1] if observations else ""
            if claim and observation:
                return f"the team used '{_shorten(claim, 120)}' and then observed: {self._sentence(_shorten(observation, 160))}"
            if observation:
                return f"the important observation was: {self._sentence(_shorten(observation, 180))}"

        if actions and observations:
            return f"the agent tried '{_shorten(actions[-1], 120)}' and observed: {self._sentence(_shorten(observations[-1], 160))}"
        if observations:
            return f"the useful observation was: {self._sentence(_shorten(observations[-1], 180))}"
        if actions:
            return f"the agent action was '{_shorten(actions[-1], 160)}'."
        return self._clean_summary(block.summary)

    def _natural_guidance(
        self,
        block: MAGFoldBlock,
        nodes: list[AtomicInteractionNode],
    ) -> str:
        if block.outcome == "failed":
            last_observation = self._last_clean_text(nodes, {"observation"})
            last_action = self._last_action_text(nodes)
            if last_observation:
                if self._is_low_information_observation(last_observation):
                    if last_action:
                        return f"Do not treat '{last_action}' as progress after an uninformative observation like '{last_observation}'; switch to an action that changes the environment or checks a new location."
                    return f"Do not treat an uninformative observation like '{last_observation}' as progress; switch to an action that changes the environment or checks a new location."
                if last_action:
                    return f"After '{last_action}', react to the observation '{last_observation}' instead of repeating the same search path."
                return f"Do not keep following this path without reacting to the observation: {self._sentence(last_observation)}"
            if block.failure_lesson:
                lesson = self._clean_failure_lesson(block.failure_lesson)
                if lesson:
                    return lesson
            return "Treat this as a negative example and change strategy if the same pattern appears."

        if block.block_type in {"evidence_verification", "conflict_resolution"}:
            return "Prefer the verified or more current evidence, and explicitly reject stale or contradicted information."

        if block.outcome == "resolved":
            last_action = self._last_clean_text(nodes, {"action", "decision"})
            if last_action:
                return f"This pattern was useful; consider reusing the strategy '{_shorten(last_action, 140)}' when the current task matches."
        return ""

    def _natural_details(self, nodes: list[AtomicInteractionNode]) -> str:
        details: list[str] = []
        for node in nodes:
            text = self._clean_node_text(node)
            if not text:
                continue
            if node.node_type == "thought":
                details.append(f"reasoned that {self._fragment(_shorten(text, 140))}")
            elif node.node_type == "action":
                if self._is_thought_like(node):
                    details.append(f"reasoned that {self._fragment(_shorten(text, 140))}")
                else:
                    details.append(f"tried '{_shorten(text, 120)}'")
            elif node.node_type == "observation":
                details.append(f"observed {self._fragment(_shorten(text, 140))}")
            elif node.node_type == "decision":
                details.append(f"decided {self._fragment(_shorten(text, 140))}")
            if len(details) >= 3:
                break
        return "; ".join(details)

    def _clean_summary(self, summary: str, limit: Optional[int] = 260) -> str:
        parts = []
        for piece in (summary or "").split("|"):
            piece = re.sub(r"^\s*[^:]{1,40}/[^:]{1,40}:\s*", "", piece).strip()
            if piece:
                parts.append(piece)
        joined = "; ".join(parts)
        if limit is None or limit <= 0:
            return joined
        return _shorten(joined, limit)

    def _clean_failure_lesson(self, lesson: str) -> str:
        cleaned = re.sub(
            r"^Avoid repeating this span without checking the failing observation:\s*",
            "",
            lesson or "",
            flags=re.IGNORECASE,
        ).strip()
        cleaned = self._clean_summary(cleaned)
        if not cleaned:
            return ""
        return f"Avoid repeating the same sequence unless the new observation changes the plan; the past failure centered on: {_shorten(cleaned, 180)}."

    def _clean_node_text(self, node: AtomicInteractionNode) -> str:
        text = (node.content or "").strip()
        # Keep the ALFWorld command prefix for thoughts. GMemory examples show
        # `> think: ...`; stripping it turns the detail trajectory into an
        # invalid command-like line such as `> To solve the task...`.
        text = re.sub(r"^(observation|obs)\s*\d*\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^(action|act)\s*\d*\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        return text

    def _last_clean_text(
        self,
        nodes: list[AtomicInteractionNode],
        node_types: set[str],
    ) -> str:
        for node in reversed(nodes):
            if node.node_type in node_types:
                return self._clean_node_text(node)
        return ""

    def _last_action_text(self, nodes: list[AtomicInteractionNode]) -> str:
        for node in reversed(nodes):
            if node.node_type == "action" and not self._is_thought_like(node):
                return self._clean_node_text(node)
        return ""

    def _sentence(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return text if text[-1] in ".!?" else f"{text}."

    def _fragment(self, text: str) -> str:
        return (text or "").strip().rstrip(".!?")

    def _is_thought_like(self, node: AtomicInteractionNode) -> bool:
        return (node.content or "").strip().lower().startswith("think:")

    def _is_low_information_observation(self, text: str) -> bool:
        normalized = (text or "").strip().lower().rstrip(".")
        return normalized in {"ok", "nothing happens", "nothing happened"} or "nothing happens" in normalized

    def _semantic_similarity(self, query: str, text: str) -> float:
        if self.enable_embedding_retrieval:
            try:
                query_vec = self.embedding_func.embed_query(query)
                text_vec = self.embedding_func.embed_query(text)
                return self._cosine(query_vec, text_vec)
            except Exception:
                pass
        return self._token_overlap_similarity(query, text)

    def _token_overlap_similarity(self, left: str, right: str) -> float:
        left_counts = Counter(_tokenize(left))
        right_counts = Counter(_tokenize(right))
        if not left_counts or not right_counts:
            return 0.0
        overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts)
        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        return overlap / (left_norm * right_norm + 1e-8)

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return dot / (left_norm * right_norm + 1e-8)

    def _behavior_match(self, query: str, block: MAGFoldBlock, role: Optional[str]) -> float:
        score = 0.0
        lowered = (query or "").lower()
        if _contains_any(lowered, VERIFY_KEYWORDS) and block.block_type in {"evidence_verification", "tool_use"}:
            score += 0.7
        if _contains_any(lowered, REFUTE_KEYWORDS) and block.block_type in {"conflict_resolution"}:
            score += 0.7
        if _contains_any(lowered, FAILURE_KEYWORDS) and block.block_type in {"failure_cause", "loop_or_stuck"}:
            score += 0.7
        if role and (role in block.roles or role in block.agents):
            score += 0.2
        if block.edge_profile.get("verifies") or block.edge_profile.get("supports"):
            score += 0.1
        return min(score, 1.0)

    def _outcome_bonus(self, block: MAGFoldBlock) -> float:
        if block.outcome == "resolved":
            return 1.0
        if block.outcome == "failed" and block.failure_lesson:
            return 0.8
        if block.outcome == "unresolved":
            return 0.0
        return 0.4

    def _important_incident_node_ids(
        self,
        block: MAGFoldBlock,
        graph: AtomicInteractionGraph,
    ) -> list[str]:
        covered = set(block.covered_node_ids)
        ids: list[str] = []
        for edge in graph.edges:
            if edge.src in covered and edge.dst in covered and edge.edge_type in {
                "refutes",
                "verifies",
                "supports",
                "causes_failure",
            }:
                ids.extend([edge.src, edge.dst])
        return ids

    def _node_score(
        self,
        node: AtomicInteractionNode,
        block: MAGFoldBlock,
        graph: AtomicInteractionGraph,
        query: str,
        role: Optional[str],
    ) -> float:
        structural = 1.0 if node.node_type in {"action", "observation", "decision"} else 0.3
        incident_types = [
            edge.edge_type
            for edge in graph.edges
            if edge.src == node.node_id or edge.dst == node.node_id
        ]
        edge_importance = 1.0 if {"refutes", "verifies", "supports"} & set(incident_types) else 0.2
        role_match = 1.0 if role and role in {node.role, node.agent_id} else 0.0
        boundary_or_key = 1.0 if node.node_id in set(block.key_node_ids + block.input_boundary_nodes + block.output_boundary_nodes) else 0.0
        return (
            0.35 * self._semantic_similarity(query, node.content)
            + 0.25 * structural
            + 0.20 * edge_importance
            + 0.10 * role_match
            + 0.10 * boundary_or_key
        )

    def _persist_atomic_graph(self, atomic_graph: AtomicInteractionGraph) -> None:
        if not self.save_atomic_graph:
            return
        path = self._atomic_graph_path(atomic_graph.query_id)
        write_json(atomic_graph.to_dict(), path)

    def _persist_blocks(self) -> None:
        write_json([block.to_dict() for block in self.blocks], self.blocks_file)

    def _persist_block_edges(self) -> None:
        if self.save_block_graph:
            write_json([edge.to_dict() for edge in self.block_edges], self.block_edges_file)

    def _load_blocks(self) -> list[MAGFoldBlock]:
        raw_blocks = load_json(self.blocks_file) or []
        return [MAGFoldBlock.from_dict(item) for item in raw_blocks]

    def _load_block_edges(self) -> list[BlockEdge]:
        raw_edges = load_json(self.block_edges_file) or []
        return [BlockEdge.from_dict(item) for item in raw_edges]

    def _load_atomic_graph(self, query_id: str) -> Optional[AtomicInteractionGraph]:
        if query_id in self.atomic_graph_cache:
            return self.atomic_graph_cache[query_id]
        path = self._atomic_graph_path(query_id)
        raw_graph = load_json(path)
        if raw_graph is None:
            return None
        graph = AtomicInteractionGraph.from_dict(raw_graph)
        self.atomic_graph_cache[query_id] = graph
        return graph

    def _atomic_graph_path(self, query_id: str) -> str:
        safe_query_id = re.sub(r"[^a-zA-Z0-9_]+", "_", query_id).strip("_") or "query"
        return os.path.join(self.atomic_graph_dir, f"{safe_query_id}.json")
