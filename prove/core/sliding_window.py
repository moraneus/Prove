"""
Sliding window graph for EPLTL verification.

Maintains the relevant portion of the global state DAG, where nodes
represent frontiers and edges represent the immediate successor relation
labeled with events. Supports backward propagation (commutation) and
redundant node removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from prove.core.event import Event
from prove.core.frontier import Frontier
from prove.core.partial_order import PartialOrder
from prove.core.summary import Summary
from prove.parser.ast_nodes import Formula


@dataclass
class GraphNode:
    """
    A node in the sliding window graph.

    Attributes:
        node_id: Unique identifier for this node.
        frontier: The frontier (global state) at this node.
        summaries: Set of EPLTL summaries for paths ending here.
        covered_processes: Processes with events after this node.
    """

    node_id: int
    frontier: Frontier
    summaries: Set[Summary] = field(default_factory=set)
    covered_processes: Set[str] = field(default_factory=set)

    def __repr__(self) -> str:
        return (
            f"GraphNode(id={self.node_id}, frontier={self.frontier}, "
            f"summaries={len(self.summaries)}, "
            f"covered={self.covered_processes})"
        )


@dataclass(frozen=True)
class GraphEdge:
    """
    An edge in the sliding window graph.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        event: The event labeling this edge.
    """

    source: int
    target: int
    event: Event


class SlidingWindowGraph:
    """
    The sliding window graph for EPLTL verification.

    Nodes represent frontiers, edges represent the immediate successor
    relation labeled with events. The graph supports backward propagation
    (commutation of independent events) and redundant node removal.

    Attributes:
        nodes: Dictionary of node_id to GraphNode.
        edges: Set of GraphEdge objects.
        maximal_node_id: ID of the current maximal frontier.
        processes: All processes in the system.
        partial_order: Complete partial order relation.
    """

    def __init__(
        self,
        processes: Iterable[str],
        initial_events: Dict[str, Event],
        formula: Formula,
        partial_order: PartialOrder,
    ) -> None:
        """
        Initialize with initial frontier and formula.

        Args:
            processes: All process IDs in the system.
            initial_events: Dictionary mapping each process to its initial event.
            formula: The EPLTL formula to verify.
            partial_order: Complete partial order for event ordering.
        """
        self.processes: frozenset[str] = frozenset(processes)
        self.formula: Formula = formula
        self.partial_order: PartialOrder = partial_order

        self.nodes: Dict[int, GraphNode] = {}
        self.edges: Set[GraphEdge] = set()
        self._next_node_id: int = 0
        self._nodes_removed: int = 0
        self._events_processed: int = 0

        # Adjacency: node_id -> list of (event, target_node_id)
        self._outgoing: Dict[int, List[Tuple[Event, int]]] = {}
        # Reverse adjacency: node_id -> list of (event, source_node_id)
        self._incoming: Dict[int, List[Tuple[Event, int]]] = {}

        # Create initial frontier from all initial events
        initial_frontier = Frontier.from_mapping(initial_events)

        # Create initial summary using global state of initial frontier
        global_props = initial_frontier.global_state()
        initial_summary = self._create_initial_summary(global_props)

        # Create initial node
        s0 = self._create_node(initial_frontier)
        s0.summaries.add(initial_summary)
        self.maximal_node_id: int = s0.node_id

    def _create_initial_summary(self, global_props: frozenset[str]) -> Summary:
        """Create the initial summary, evaluated against the initial global state."""
        s = Summary.initial(self.formula)
        # Update with a synthetic event carrying the global state propositions
        dummy = Event(
            eid="__init__",
            process="__init__",
            vector_clock=None,  # type: ignore[arg-type]
            timestamp=0.0,
            propositions=global_props,
        )
        return s.update(dummy, self.formula)

    def _create_node(self, frontier: Frontier) -> GraphNode:
        """Create a new graph node with the given frontier."""
        nid = self._next_node_id
        self._next_node_id += 1
        node = GraphNode(node_id=nid, frontier=frontier)
        self.nodes[nid] = node
        self._outgoing[nid] = []
        self._incoming[nid] = []
        return node

    def _add_edge(
        self, source_id: int, event: Event, target_id: int,
        new_edges: Optional[List[GraphEdge]] = None,
    ) -> None:
        """Add an edge from source to target labeled with event."""
        edge = GraphEdge(source=source_id, target=target_id, event=event)
        self.edges.add(edge)
        self._outgoing[source_id].append((event, target_id))
        self._incoming[target_id].append((event, source_id))
        if new_edges is not None:
            new_edges.append(edge)

    def _has_edge_with_event(self, source_id: int, event: Event) -> Optional[int]:
        """Return target node ID if source has an outgoing edge with this event."""
        for ev, tid in self._outgoing.get(source_id, []):
            if ev is event:
                return tid
        return None

    def _find_node_with_frontier(self, frontier: Frontier) -> Optional[int]:
        """Find an existing node with the given frontier, or None."""
        for nid, node in self.nodes.items():
            if node.frontier == frontier:
                return nid
        return None

    def _can_commute(self, new_event: Event, edge_event: Event) -> bool:
        """
        Check if new_event can commute backward over an edge labeled edge_event.

        Paper V2 conditions:
        1. Events are independent: pr(new_event) != pr(edge_event)
        2. Timing allows: NOT (t(new_event) - t(edge_event) > epsilon)
        """
        # Condition 1: process independence
        if new_event.process == edge_event.process:
            return False
        # Condition 2: timing constraint
        if (new_event.timestamp - edge_event.timestamp) > self.partial_order.epsilon:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Main event processing
    # ------------------------------------------------------------------ #

    def process_event(self, event: Event) -> None:
        """
        Process a new event, updating the graph.

        Steps:
        1. Add edge from maximal node to new node
        2. Backward propagation (commutation)
        3. Update summaries for all new edges
        4. Update covered processes
        5. Remove redundant nodes
        """
        self._events_processed += 1
        new_edges: List[GraphEdge] = []

        # Step 1: Create new maximal node
        old_max_id = self.maximal_node_id
        old_max_node = self.nodes[old_max_id]
        new_frontier = old_max_node.frontier.successor(event, self.partial_order)
        new_max_node = self._create_node(new_frontier)
        self._add_edge(old_max_id, event, new_max_node.node_id, new_edges)

        # Step 2: Backward propagation (also collects new edges)
        self._backward_propagate(event, old_max_id, new_edges)

        # Step 3: Propagate summaries through all new edges (worklist)
        self._propagate_summaries(new_edges)

        # Step 4: Update maximal node
        self.maximal_node_id = new_max_node.node_id

        # Step 5: Update covered processes
        self._update_covered_processes(event)

        # Step 6: Remove redundant nodes
        self._remove_redundant_nodes()

    def _backward_propagate(
        self, event: Event, start_id: int, new_edges: List[GraphEdge]
    ) -> None:
        """
        Propagate event backward over independent edges (DFS).

        For each incoming edge (s'' --f--> s') where f and event are
        independent (can commute), create the commuted diamond.
        """
        visited: Set[int] = {start_id}
        stack: List[int] = [start_id]

        while stack:
            current_id = stack.pop()

            for edge_event, source_id in list(self._incoming.get(current_id, [])):
                if not self._can_commute(event, edge_event):
                    continue

                # Check if source already has an outgoing edge with this event
                existing_target = self._has_edge_with_event(source_id, event)
                if existing_target is None:
                    # Create the commuted path: source --event--> r --edge_event--> join
                    source_node = self.nodes[source_id]
                    r_frontier = source_node.frontier.successor(
                        event, self.partial_order
                    )

                    # Find or create the intermediate node
                    r_id = self._find_node_with_frontier(r_frontier)
                    if r_id is None:
                        r_node = self._create_node(r_frontier)
                        r_id = r_node.node_id

                    self._add_edge(source_id, event, r_id, new_edges)

                    # Find the join node (where both paths meet)
                    r_node_obj = self.nodes[r_id]
                    join_frontier = r_node_obj.frontier.successor(
                        edge_event, self.partial_order
                    )
                    join_id = self._find_node_with_frontier(join_frontier)
                    if join_id is None:
                        join_node = self._create_node(join_frontier)
                        join_id = join_node.node_id

                    # Add edge from r to join (if not exists)
                    if self._has_edge_with_event(r_id, edge_event) is None:
                        self._add_edge(r_id, edge_event, join_id, new_edges)

                if source_id not in visited:
                    visited.add(source_id)
                    stack.append(source_id)

    def _propagate_summaries(self, new_edges: List[GraphEdge]) -> None:
        """
        Propagate summaries through all new edges using a worklist.

        Repeatedly processes edges whose source nodes have summaries,
        until no new summaries are generated. This handles both the
        current event's edges and cross-edges from backward propagation.
        """
        # Build a worklist of edges to process
        worklist = list(new_edges)
        changed = True

        while changed:
            changed = False
            next_worklist: List[GraphEdge] = []

            for edge in worklist:
                source_node = self.nodes.get(edge.source)
                target_node = self.nodes.get(edge.target)
                if source_node is None or target_node is None:
                    continue
                if not source_node.summaries:
                    # Source has no summaries yet; retry later
                    next_worklist.append(edge)
                    continue

                target_global = target_node.frontier.global_state()
                state_event = Event(
                    eid=edge.event.eid,
                    process=edge.event.process,
                    vector_clock=edge.event.vector_clock,
                    timestamp=edge.event.timestamp,
                    propositions=target_global,
                )

                for summary in list(source_node.summaries):
                    new_summary = summary.update(state_event, self.formula)
                    if new_summary not in target_node.summaries:
                        target_node.summaries.add(new_summary)
                        changed = True

            worklist = next_worklist

    def _update_covered_processes(self, event: Event) -> None:
        """
        Update covered process sets after processing an event.

        For every node in the graph, add the event's process to the
        covered set (since this process now has an event beyond that node).
        """
        proc = event.process
        for node in self.nodes.values():
            if node.node_id != self.maximal_node_id:
                node.covered_processes.add(proc)

    def _remove_redundant_nodes(self) -> None:
        """
        Remove nodes where all processes are covered.

        A node is redundant when R_s = P (all processes have events
        after this node), meaning it can never affect the final verdict.
        """
        to_remove: List[int] = []
        for nid, node in self.nodes.items():
            if nid == self.maximal_node_id:
                continue
            if node.covered_processes >= self.processes:
                to_remove.append(nid)

        for nid in to_remove:
            self._remove_node(nid)
            self._nodes_removed += 1

    def _remove_node(self, node_id: int) -> None:
        """Remove a node and all its edges from the graph."""
        # Remove all edges involving this node
        edges_to_remove = {
            e for e in self.edges if e.source == node_id or e.target == node_id
        }
        self.edges -= edges_to_remove

        # Clean up adjacency lists
        for ev, tid in self._outgoing.get(node_id, []):
            self._incoming[tid] = [
                (e, s) for e, s in self._incoming.get(tid, []) if s != node_id
            ]
        for ev, sid in self._incoming.get(node_id, []):
            self._outgoing[sid] = [
                (e, t) for e, t in self._outgoing.get(sid, []) if t != node_id
            ]

        del self._outgoing[node_id]
        del self._incoming[node_id]
        del self.nodes[node_id]

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #

    def get_verdict(
        self, formula: Formula
    ) -> Tuple[bool, Optional[List[Event]]]:
        """
        Check if formula is satisfied at maximal frontier.

        The property is satisfied (existentially) if at least one
        summary at the maximal node evaluates the formula to True.

        Returns:
            (is_satisfied, witness_path or None)
        """
        max_node = self.nodes[self.maximal_node_id]
        for summary in max_node.summaries:
            if summary.evaluate(formula):
                return True, None
        return False, None

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the graph."""
        max_summaries = max(
            (len(n.summaries) for n in self.nodes.values()), default=0
        )
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes_removed": self._nodes_removed,
            "max_summaries": max_summaries,
            "events_processed": self._events_processed,
        }
