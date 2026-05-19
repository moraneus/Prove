#!/usr/bin/env python3
"""
RACE-style airspace-monitoring trace generator for PROVE experiments.

Generates valid CSV traces matching the structure used by
``experiments/Safety_Race``. Each aircraft emits a periodic
``pos_update`` event together with the ``safe_AiAj`` propositions
covering every pairwise mutual-separation guarantee involving that
aircraft. There are no messages between aircraft; each event is a
local position update.

Trace layout:
    1. Five init events at t=0.0 (one per aircraft, A1..A5).
    2. Periodic ticks at t=1.0, 2.0, 3.0, ...; each tick contributes
       5 events (one per aircraft, all sharing the tick's timestamp).
    3. A trailing partial tick contributes 1-4 events at the next
       integer timestamp when the requested size does not divide evenly.

Per-event propositions (canonical ordering, lower index first):
    A1: pos_update | safe_A1A2 | safe_A1A3 | safe_A1A4 | safe_A1A5
    A2: pos_update | safe_A1A2 | safe_A2A3 | safe_A2A4 | safe_A2A5
    A3: pos_update | safe_A1A3 | safe_A2A3 | safe_A3A4 | safe_A3A5
    A4: pos_update | safe_A1A4 | safe_A2A4 | safe_A3A4 | safe_A4A5
    A5: pos_update | safe_A1A5 | safe_A2A5 | safe_A3A5 | safe_A4A5

Because the trace contains no ``unsafe_AiAj`` propositions, the
mutual-separation property
        !(TRUE S unsafe_A1A2) & !(TRUE S unsafe_A1A3) & !(TRUE S unsafe_A2A3)
is satisfied vacuously at every ε.

Usage:
    python tools/generate_race_trace.py --events 1000 \\
        --output experiments/Safety_Race/trace_race.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROCESSES = ("A1", "A2", "A3", "A4", "A5")
EVENTS_PER_TICK = len(PROCESSES)  # one event per aircraft per tick
TICK_PERIOD = 1.0  # mean seconds between consecutive ticks
TICK_PERIOD_JITTER = 0.1  # ± jitter on tick period (clamped to stay positive)
TICK_INTRA_SPREAD = 0.05  # ± jitter applied to each aircraft within a tick
EPSILON = 1.0  # default clock-skew bound carried in the header
MIN_EVENTS = len(PROCESSES)  # 5 init events


# ---------------------------------------------------------------------------
# Proposition layout
# ---------------------------------------------------------------------------


def safe_props_for(proc: str) -> str:
    """Return ``pos_update`` plus every ``safe_AiAj`` pair involving ``proc``,
    using canonical ordering (lower index first)."""
    i = int(proc[1:])
    pair_props: list[str] = []
    for q in PROCESSES:
        if q == proc:
            continue
        j = int(q[1:])
        a, b = (proc, q) if i < j else (q, proc)
        pair_props.append(f"safe_{a}{b}")
    return "|".join(["pos_update", *pair_props])


# ---------------------------------------------------------------------------
# Generator state
# ---------------------------------------------------------------------------


class GeneratorState:
    """Tracks per-aircraft event counters and accumulates the event list.

    No vector-clock merges occur because Race events are purely local --
    each aircraft only increments its own component.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {p: 0 for p in PROCESSES}
        self.events: list[tuple[str, str, str, str, str, str, str]] = []

    def _vc_str(self, proc: str) -> str:
        return ";".join(
            f"{p}:{self.counts[p] if p == proc else 0}" for p in PROCESSES
        )

    def emit(self, proc: str, timestamp: float) -> None:
        self.counts[proc] += 1
        eid = f"{proc.lower()}_{self.counts[proc]}"
        ts = f"{timestamp:.3f}"
        self.events.append(
            (
                eid,
                proc,
                self._vc_str(proc),
                ts,
                safe_props_for(proc),
                "local",
                "",
            )
        )


# ---------------------------------------------------------------------------
# Trace assembly
# ---------------------------------------------------------------------------


def emit_init(state: GeneratorState) -> None:
    for p in PROCESSES:
        state.emit(p, 0.0)


def _jitter_tick_times(rng: random.Random, num_ticks: int) -> list[float]:
    """Cumulative tick start times with jittered periods (always strictly
    increasing and >= TICK_PERIOD/4 apart)."""
    times: list[float] = []
    t = 0.0
    for _ in range(num_ticks):
        period = max(
            TICK_PERIOD / 4,
            TICK_PERIOD + rng.uniform(-TICK_PERIOD_JITTER, TICK_PERIOD_JITTER),
        )
        t += period
        times.append(t)
    return times


def emit_trace(target_size: int, seed: int | None) -> GeneratorState:
    if target_size < MIN_EVENTS:
        raise ValueError(
            f"trace size must be at least {MIN_EVENTS} (one init event per aircraft)"
        )

    rng = random.Random(seed)

    state = GeneratorState()
    emit_init(state)

    remaining = target_size - MIN_EVENTS
    full_ticks, partial = divmod(remaining, EVENTS_PER_TICK)
    total_ticks = full_ticks + (1 if partial else 0)
    tick_times = _jitter_tick_times(rng, total_ticks)

    for tick_idx in range(full_ticks):
        base = tick_times[tick_idx]
        for proc in PROCESSES:
            t = base + rng.uniform(-TICK_INTRA_SPREAD, TICK_INTRA_SPREAD)
            state.emit(proc, max(0.0, t))

    if partial:
        base = tick_times[full_ticks]
        for proc in PROCESSES[:partial]:
            t = base + rng.uniform(-TICK_INTRA_SPREAD, TICK_INTRA_SPREAD)
            state.emit(proc, max(0.0, t))

    assert len(state.events) == target_size, (
        f"internal error: emitted {len(state.events)} events, "
        f"expected {target_size}"
    )
    return state


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


HEADER_TEMPLATE = """\
# Generated RACE-style airspace monitoring trace
# total_events: {total}
# process_count: {nproc}
# system_processes: {procs}
# epsilon: {epsilon}
# Full ticks: {full_ticks} ({events_per_tick} events each, ~{tick_period}s apart with jitter)
# Partial tick events: {partial}
# Random seed: {seed}
# Property: pairwise !(TRUE S unsafe_AiAj)  (no unsafe events emitted)
"""


def write_trace(state: GeneratorState, output: Path, seed: int | None) -> None:
    remaining = len(state.events) - MIN_EVENTS
    full_ticks, partial = divmod(remaining, EVENTS_PER_TICK)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        f.write(
            HEADER_TEMPLATE.format(
                total=len(state.events),
                nproc=len(PROCESSES),
                procs="|".join(PROCESSES),
                epsilon=EPSILON,
                full_ticks=full_ticks,
                events_per_tick=EVENTS_PER_TICK,
                tick_period=TICK_PERIOD,
                partial=partial,
                seed="random" if seed is None else seed,
            )
        )
        w = csv.writer(f)
        w.writerow(
            [
                "eid",
                "processes",
                "vc",
                "timestamp",
                "props",
                "event_type",
                "msg_partner",
            ]
        )
        w.writerows(state.events)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Race-style airspace trace of the requested length.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Each tick contributes {EVENTS_PER_TICK} events (one per\n"
            f"aircraft) at the same timestamp. The trace is assembled as\n"
            f"{MIN_EVENTS} init events + k full ticks + a partial tick whose\n"
            "length completes the requested total."
        ),
    )
    parser.add_argument(
        "--events",
        "-n",
        type=int,
        required=True,
        help=f"target number of events (>= {MIN_EVENTS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="output CSV path (parent directories are created as needed)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="random seed for jittered timing (default: time-based)",
    )
    args = parser.parse_args(argv)

    try:
        state = emit_trace(args.events, args.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    write_trace(state, args.output, args.seed)

    remaining = args.events - MIN_EVENTS
    full_ticks, partial = divmod(remaining, EVENTS_PER_TICK)
    print(
        f"wrote {args.events} events to {args.output} "
        f"({MIN_EVENTS} init + {full_ticks} full tick × "
        f"{EVENTS_PER_TICK} + {partial} partial)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
