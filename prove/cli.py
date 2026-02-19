"""
Command-line interface for the PROVE runtime verification engine.

Provides argument parsing and orchestration for verifying EPLTL properties
over partial order executions supplied as CSV trace files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import prove
from prove.core.monitor import EPLTLMonitor
from prove.parser.formula import parse_formula
from prove.utils.logger import LogLevel, MonitorLogger
from prove.utils.trace_reader import TraceReader
from prove.utils.visualization import GraphVisualizer, PartialOrderVisualizer


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the PROVE CLI."""
    parser = argparse.ArgumentParser(
        prog="prove",
        description=(
            "PROVE: Partial oRder Verification Engine - "
            "Runtime verification of EPLTL properties "
            "over partial order executions"
        ),
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-p",
        "--property",
        type=Path,
        required=True,
        help="Path to property file (.prop) containing EPLTL formula",
    )
    required.add_argument(
        "-t",
        "--trace",
        type=Path,
        required=True,
        help="Path to trace file (.csv)",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=None,
        help="Maximum clock skew (epsilon) in time units (default: infinity)",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["silent", "normal", "verbose"],
        default="normal",
        help="Output level (default: normal)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Debug level 0-3 (default: 0)",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const="__stdout__",
        default=None,
        metavar="FILE",
        help="Generate graph visualization (optionally to FILE)",
    )
    parser.add_argument(
        "--visualize-ascii",
        action="store_true",
        help="Print ASCII graph visualization to terminal",
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Keep all graph nodes for visualization (disables pruning)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics after verification",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"prove {prove.__version__}",
    )

    return parser


def _resolve_log_level(output: str, debug: int) -> LogLevel:
    """Determine the effective log level from output and debug settings."""
    if debug >= 3:
        return LogLevel.DEBUG
    if debug >= 2:
        return LogLevel.VERBOSE
    if output == "verbose" or debug >= 1:
        return LogLevel.VERBOSE
    if output == "silent":
        return LogLevel.SILENT
    return LogLevel.NORMAL


def main() -> None:
    """Entry point for the ``prove`` CLI command."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _run(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


def _run(args: argparse.Namespace) -> None:
    """Execute the verification pipeline."""
    # Validate input files exist
    if not args.property.exists():
        print(f"Error: Property file not found: {args.property}", file=sys.stderr)
        sys.exit(2)

    if not args.trace.exists():
        print(f"Error: Trace file not found: {args.trace}", file=sys.stderr)
        sys.exit(2)

    # Set up logger
    log_level = _resolve_log_level(args.output, args.debug)
    stream = sys.stdout if args.output != "silent" else open("/dev/null", "w")
    logger = MonitorLogger(level=log_level, stream=stream)

    # Parse property file
    prop_text = args.property.read_text().strip()
    formula_lines = [
        line for line in prop_text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if not formula_lines:
        print("Error: Property file is empty", file=sys.stderr)
        sys.exit(2)
    formula_str = " ".join(formula_lines)
    formula = parse_formula(formula_str)

    # Read trace
    reader = TraceReader(args.trace)
    trace_data = reader.read_all(epsilon=args.epsilon)

    eps: float
    if args.epsilon is not None:
        eps = args.epsilon
    elif trace_data.metadata.epsilon is not None:
        eps = trace_data.metadata.epsilon
    else:
        eps = float("inf")

    # Create and run monitor
    monitor = EPLTLMonitor(
        formula=formula,
        processes=trace_data.metadata.processes,
        epsilon=eps,
        logger=logger,
        full_graph=args.full_graph,
    )
    result = monitor._run_with_partial_order(trace_data.events, trace_data.partial_order)

    # ASCII visualization
    if args.visualize_ascii:
        po_viz = PartialOrderVisualizer(trace_data.partial_order, epsilon=eps)
        print()
        print(po_viz.render())

    # DOT/PNG visualization
    if args.visualize is not None:
        graph = monitor._graph
        if graph is not None:
            viz = GraphVisualizer(graph)
            # Use step-by-step DOT when --full-graph is set
            dot_content = viz.to_dot_steps() if args.full_graph else viz.to_dot()
            if args.visualize == "__stdout__":
                print(dot_content)
            else:
                filepath = Path(args.visualize)
                suffix = filepath.suffix.lower()
                if suffix == ".dot":
                    filepath.write_text(dot_content)
                elif suffix in (".png", ".pdf", ".svg") and not args.full_graph:
                    try:
                        viz.save_png(filepath)
                    except RuntimeError as e:
                        print(f"Warning: {e}", file=sys.stderr)
                        dot_path = filepath.with_suffix(".dot")
                        dot_path.write_text(dot_content)
                else:
                    filepath.write_text(dot_content)

    # Statistics (skip if verbose already printed them)
    if args.stats and log_level.value < LogLevel.VERBOSE.value:
        print()
        print("=== Statistics ===")
        for key, value in result.statistics.items():
            label = key.replace("_", " ").title()
            print(f"  {label}: {value}")

    # Exit with appropriate code
    if result.satisfied:
        sys.exit(0)
    else:
        sys.exit(1)
