"""
Visualization utilities for PROVE.

Generates visual representations of the sliding window graph (DOT,
ASCII, JSON) and ASCII timeline diagrams of partial order executions
showing process columns, timestamps, and cross-process orderings.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prove.core.event import Event
from prove.core.partial_order import PartialOrder
from prove.core.sliding_window import SlidingWindowGraph


class GraphVisualizer:
    """
    Visualizer for sliding window graphs.

    Renders the sliding window graph (nodes = frontiers, edges = events)
    in DOT (Graphviz), ASCII, and JSON formats.

    Attributes:
        graph: The sliding window graph to visualize.
    """

    def __init__(self, graph: SlidingWindowGraph) -> None:
        """
        Initialize with a sliding window graph.

        Args:
            graph: The SlidingWindowGraph to visualize.
        """
        self.graph = graph

    def to_dot(self) -> str:
        """
        Generate DOT format string for Graphviz rendering.

        Nodes are labeled with their frontier events, edges are labeled
        with the event that transitions between frontiers.

        Returns:
            A DOT format string.
        """
        lines: List[str] = ["digraph SlidingWindowGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=filled, fillcolor=lightyellow];")

        for nid, node in self.graph.nodes.items():
            events_str = ", ".join(sorted(e.eid for e in node.frontier.events))
            n_summaries = len(node.summaries)
            label = f"F{nid}\\n{{{events_str}}}\\n({n_summaries} summaries)"
            style = ""
            if nid == self.graph.maximal_node_id:
                style = ", fillcolor=lightblue, penwidth=2"
            lines.append(f'  n{nid} [label="{label}"{style}];')

        for edge in self.graph.edges:
            lines.append(f"  n{edge.source} -> n{edge.target} " f'[label="{edge.event.eid}"];')

        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self, max_width: int = 80) -> str:
        """
        Generate ASCII art representation of the sliding window graph.

        Shows nodes and edges in a top-down layout.

        Args:
            max_width: Maximum line width.

        Returns:
            ASCII art string.
        """
        lines: List[str] = []
        lines.append("=== Sliding Window Graph ===")
        lines.append("")

        # Collect nodes layer by layer (BFS from roots)
        node_ids = sorted(self.graph.nodes.keys())
        for nid in node_ids:
            node = self.graph.nodes[nid]
            events_str = ", ".join(sorted(e.eid for e in node.frontier.events))
            marker = " *" if nid == self.graph.maximal_node_id else ""
            label = f"[F{nid}] {{{events_str}}} ({len(node.summaries)} sum){marker}"
            if len(label) > max_width:
                label = label[: max_width - 3] + "..."
            lines.append(label)

            # Show outgoing edges
            for ev, tid in self.graph._outgoing.get(nid, []):
                edge_label = f"  --{ev.eid}--> [F{tid}]"
                if len(edge_label) > max_width:
                    edge_label = edge_label[: max_width - 3] + "..."
                lines.append(edge_label)

        return "\n".join(lines)

    def to_json(self) -> str:
        """
        Generate JSON representation of the graph.

        Returns:
            A JSON string with nodes and edges.
        """
        nodes: List[Dict[str, Any]] = []
        for nid, node in self.graph.nodes.items():
            nodes.append(
                {
                    "id": nid,
                    "frontier": sorted(e.eid for e in node.frontier.events),
                    "summaries": len(node.summaries),
                    "covered_processes": sorted(node.covered_processes),
                    "is_maximal": nid == self.graph.maximal_node_id,
                }
            )

        edges: List[Dict[str, Any]] = []
        for edge in self.graph.edges:
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "event": edge.event.eid,
                }
            )

        return json.dumps({"nodes": nodes, "edges": edges}, indent=2)

    def save_dot(self, filepath: Path) -> None:
        """
        Save DOT format to a file.

        Args:
            filepath: Path to write the DOT file.
        """
        filepath.write_text(self.to_dot())

    def save_png(self, filepath: Path) -> None:
        """
        Render graph to PNG using Graphviz.

        Falls back gracefully if Graphviz is not installed.

        Args:
            filepath: Path to write the PNG file.
        """
        dot_content = self.to_dot()
        try:
            result = subprocess.run(
                ["dot", "-Tpng", "-o", str(filepath)],
                input=dot_content,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Graphviz error: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Graphviz 'dot' command not found. " "Install Graphviz to render PNG files."
            )


class PartialOrderVisualizer:
    """
    ASCII visualizer for partial order executions.

    Renders a timeline-style diagram showing:
    - One column per process, events aligned vertically by time
    - Timestamps shown below each event name
    - Vertical arrows for intra-process ordering
    - Cross-process ordering annotations
    - Summary of all cross-process orderings at the bottom

    Attributes:
        partial_order: The computed partial order over events.
        epsilon: Clock skew parameter (for annotation).
    """

    def __init__(
        self,
        partial_order: PartialOrder,
        epsilon: float = float("inf"),
    ) -> None:
        """
        Initialize with partial order and epsilon.

        Args:
            partial_order: The computed partial order over events.
            epsilon: Clock skew parameter (for annotation).
        """
        self.partial_order = partial_order
        self.epsilon = epsilon

    def render(self, max_width: int = 120) -> str:
        """
        Generate the full ASCII diagram string.

        Layout:
        1. Header with process names and epsilon
        2. Events in timestamp order, placed in process columns
        3. Intra-process vertical arrows
        4. Inline cross-process ordering annotations between groups
        5. Propositions shown next to event names
        6. Footer listing all cross-process orderings

        Args:
            max_width: Maximum line width.

        Returns:
            Multi-line string ready for terminal output.
        """
        events = list(self.partial_order._events)
        if not events:
            return "(no events)"

        # Group events by process, sort by timestamp
        by_process: Dict[str, List[Event]] = defaultdict(list)
        for e in events:
            by_process[e.process].append(e)

        processes = sorted(by_process.keys())
        for p in processes:
            by_process[p].sort(key=lambda ev: (ev.timestamp, ev.eid))

        # Compute column width
        n_procs = len(processes)
        col_width = max(16, min(max_width // max(n_procs, 1), 24))

        proc_col = {p: i for i, p in enumerate(processes)}

        # Collect cross-process orderings
        cross_orderings = self._find_cross_process_orderings(events)

        # Build lookup: target event eid -> list of (source, target, reason)
        incoming_orderings: Dict[str, List[Tuple[Event, Event, str]]] = defaultdict(list)
        for src, tgt, reason in cross_orderings:
            incoming_orderings[tgt.eid].append((src, tgt, reason))

        # Build output lines
        lines: List[str] = []

        # Header
        header = ""
        for p in processes:
            header += p.center(col_width)
        lines.append(header.rstrip())

        separator = "─" * min(len(header), max_width)
        lines.append(separator)

        # Epsilon line
        eps_str = "inf" if self.epsilon == float("inf") else f"{self.epsilon}"
        lines.append(f"(ε = {eps_str})")
        lines.append("")

        # Build all events sorted by timestamp then by process
        all_sorted: List[Event] = []
        for p in processes:
            all_sorted.extend(by_process[p])
        all_sorted.sort(key=lambda ev: (ev.timestamp, processes.index(ev.process)))

        # Track last event per process (for intra-process arrows)
        last_event_per_proc: Dict[str, Optional[Event]] = {p: None for p in processes}

        # Group events by timestamp for same-row rendering
        ts_groups: List[Tuple[float, List[Event]]] = []
        if all_sorted:
            current_ts = all_sorted[0].timestamp
            current_group: List[Event] = [all_sorted[0]]
            for ev in all_sorted[1:]:
                if ev.timestamp == current_ts:
                    current_group.append(ev)
                else:
                    ts_groups.append((current_ts, current_group))
                    current_ts = ev.timestamp
                    current_group = [ev]
            ts_groups.append((current_ts, current_group))

        first_group = True
        for _ts, group in ts_groups:
            # Collect cross-process annotations targeting events in this group
            group_annotations: List[Tuple[Event, Event, str]] = []
            for e in group:
                group_annotations.extend(incoming_orderings.get(e.eid, []))

            # Intra-process arrows before this row (if not first)
            if not first_group:
                arrow_line = ""
                for p in processes:
                    has_event_in_group = any(e.process == p for e in group)
                    had_prev = last_event_per_proc[p] is not None
                    if had_prev and has_event_in_group:
                        arrow_line += "│".center(col_width)
                    else:
                        arrow_line += " " * col_width
                stripped = arrow_line.rstrip()
                if stripped.strip():
                    lines.append(stripped)

                arrow_line2 = ""
                for p in processes:
                    has_event_in_group = any(e.process == p for e in group)
                    had_prev = last_event_per_proc[p] is not None
                    if had_prev and has_event_in_group:
                        arrow_line2 += "↓".center(col_width)
                    else:
                        arrow_line2 += " " * col_width
                stripped2 = arrow_line2.rstrip()
                if stripped2.strip():
                    lines.append(stripped2)

            # Inline cross-process arrows before the event name
            for src, tgt, reason in group_annotations:
                src_col = proc_col[src.process]
                tgt_col = proc_col[tgt.process]
                annotation = self._render_cross_arrow(
                    src,
                    tgt,
                    reason,
                    src_col,
                    tgt_col,
                    col_width,
                    n_procs,
                )
                lines.append(annotation)

            first_group = False

            # Event name + propositions line
            name_line = ""
            for p in processes:
                evts_in_col = [e for e in group if e.process == p]
                if evts_in_col:
                    ev = evts_in_col[0]
                    if ev.propositions:
                        props = ", ".join(sorted(ev.propositions))
                        label = f"{ev.eid}  {{{props}}}"
                    else:
                        label = ev.eid
                    name_line += label.center(col_width)
                else:
                    name_line += " " * col_width
            lines.append(name_line.rstrip())

            # Timestamp line
            ts_line = ""
            for p in processes:
                evts_in_col = [e for e in group if e.process == p]
                if evts_in_col:
                    ts_line += f"(t={evts_in_col[0].timestamp})".center(col_width)
                else:
                    ts_line += " " * col_width
            lines.append(ts_line.rstrip())

            # Update last event per process
            for e in group:
                last_event_per_proc[e.process] = e

        # Footer: cross-process orderings
        lines.append("")
        lines.append(separator)

        if cross_orderings:
            lines.append("Cross-process orderings:")
            for src, tgt, reason in cross_orderings:
                entry = f"  {src.eid} ≺ {tgt.eid}  ({reason})"
                if len(entry) > max_width:
                    entry = entry[: max_width - 3] + "..."
                lines.append(entry)
        else:
            lines.append("No cross-process orderings.")

        return "\n".join(lines)

    def _render_cross_arrow(
        self,
        src: Event,
        tgt: Event,
        reason: str,
        src_col: int,
        tgt_col: int,
        col_width: int,
        n_procs: int,
    ) -> str:
        """
        Render an inline cross-process arrow annotation.

        Draws a horizontal arrow from the source column to the target
        column, annotated with the ordering reason.

        Args:
            src: Source event.
            tgt: Target event.
            reason: Ordering reason string.
            src_col: Column index of source process.
            tgt_col: Column index of target process.
            col_width: Width of each process column.
            n_procs: Number of processes.

        Returns:
            Formatted annotation line.
        """
        total_width = col_width * n_procs
        line = list(" " * total_width)

        # Column center positions
        src_center = src_col * col_width + col_width // 2
        tgt_center = tgt_col * col_width + col_width // 2

        left = min(src_center, tgt_center)
        right = max(src_center, tgt_center)

        # Fill the arrow span
        for i in range(left, right + 1):
            if i < total_width:
                line[i] = "─"

        # Place endpoints and direction
        if src_col < tgt_col:
            if src_center < total_width:
                line[src_center] = "╰"
            if tgt_center < total_width:
                line[tgt_center] = "→"
        else:
            if tgt_center < total_width:
                line[tgt_center] = "←"
            if src_center < total_width:
                line[src_center] = "╯"

        # Place the annotation label in the middle
        label = f" {src.eid} ≺ {tgt.eid} ({reason}) "
        mid = (left + right) // 2
        label_start = mid - len(label) // 2
        # Clamp within the arrow span
        label_start = max(left + 1, min(label_start, right - len(label)))
        if label_start < 0:
            label_start = 0
        for i, ch in enumerate(label):
            pos = label_start + i
            if 0 <= pos < total_width:
                line[pos] = ch

        return "".join(line).rstrip()

    def _find_cross_process_orderings(self, events: List[Event]) -> List[Tuple[Event, Event, str]]:
        """
        Find all direct cross-process orderings.

        Returns tuples of (source, target, reason_string) for orderings
        where source and target are on different processes and there is
        no intermediate event between them (immediate predecessors only).
        """
        orderings: List[Tuple[Event, Event, str]] = []

        for e in events:
            imm_preds = self.partial_order.immediate_predecessors(e)
            for pred in imm_preds:
                if pred.process != e.process:
                    reason = self._classify_ordering_reason(pred, e)
                    orderings.append((pred, e, reason))

        return orderings

    def _classify_ordering_reason(self, e1: Event, e2: Event) -> str:
        """
        Determine why e1 ≺ e2 for annotation purposes.

        Priority:
        1. If e1 is a send and e2 is the matching receive -> "VC"
        2. If VC(e1) < VC(e2) -> "VC"
        3. If timestamp(e2) - timestamp(e1) > epsilon -> "Δt=X.X>ε=Y.Y"
        4. If both apply -> show both

        Args:
            e1: Source event.
            e2: Target event.

        Returns:
            A reason string like "VC" or "Δt=4.0>ε=2.0".
        """
        vc_ordered = (
            e1.vector_clock is not None
            and e2.vector_clock is not None
            and e1.vector_clock < e2.vector_clock
        )

        time_diff = e2.timestamp - e1.timestamp
        eps_ordered = self.epsilon != float("inf") and time_diff > self.epsilon

        if vc_ordered and eps_ordered:
            return f"VC, Δt={time_diff:.1f}>ε={self.epsilon:.1f}"
        elif vc_ordered:
            return "VC"
        elif eps_ordered:
            return f"Δt={time_diff:.1f}>ε={self.epsilon:.1f}"
        else:
            return "VC"

    def _format_cross_process_edge(self, source: Event, target: Event, reason: str) -> str:
        """
        Format a single cross-process ordering annotation.

        Args:
            source: Source event (e where e ≺ f).
            target: Target event (f where e ≺ f).
            reason: Ordering reason string.

        Returns:
            Formatted annotation string.
        """
        return f"{source.eid} ≺ {target.eid}  ({reason})"
