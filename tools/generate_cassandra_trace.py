#!/usr/bin/env python3
"""
Cassandra-style distributed-DB trace generator for PROVE experiments.

Generates valid CSV traces with correct vector clocks, FIFO message
ordering, and the write-reference-check workload from
``experiments/Safety_Cassandra``.

Topology: five replicas P1..P5 connected in a ring. Each round
propagates a STUDENT row through the ring (P1 → P2 → P3 → P4 → P5),
where every replica applies the row locally; P5 then issues an
ENROLLMENT row referencing the student and ships it back to P1, which
applies it. Replicas exchange asynchronous FIFO messages.

Per-round event layout (17 events, round T starts at 1.0+(N-1)*10.6):
    1   P1.write_student      (local,  label=write_student)     T+0.0
    2   P1.send_student       (P1→P2 send, label=send_student)  T+0.1
    3   P2.recv_student       (recv,   label=recv_student)      T+0.9
    4   P2.apply_student      (local,  label=apply_student)     T+1.1
    5   P2.forward_student    (P2→P3 send, label=forward_student) T+1.5
    6   P3.recv_student       (recv,   label=recv_student)      T+2.3
    7   P3.apply_student      (local,  label=apply_student)     T+2.5
    8   P3.forward_student    (P3→P4 send, label=forward_student) T+2.9
    9   P4.recv_student       (recv,   label=recv_student)      T+3.7
    10  P4.apply_student      (local,  label=apply_student)     T+3.9
    11  P4.forward_student    (P4→P5 send, label=forward_student) T+4.3
    12  P5.recv_student       (recv,   label=recv_student)      T+5.1
    13  P5.apply_student      (local,  label=apply_student)     T+5.3
    14  P5.write_enrollment   (local,  label=write_enrollment)  T+6.1
    15  P5.send_enrollment    (P5→P1 send, label=send_enrollment) T+6.3
    16  P1.recv_enrollment    (recv,   label=recv_enrollment)   T+7.1
    17  P1.apply_enrollment   (local,  label=apply_enrollment)  T+7.3

The reference-check property
        write_enrollment → (TRUE S write_student)
is causally satisfied by the chain above, so every linearization
preserves it for any ε.

Usage:
    python tools/generate_cassandra_trace.py --events 1000 \\
        --output experiments/Safety_Cassandra/trace_cass.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Callable

PROCESSES = ("P1", "P2", "P3", "P4", "P5")
EVENTS_PER_ROUND = 17
ROUND_START = 1.0  # first round begins at this timestamp
ROUND_PERIOD = 10.6  # mean seconds between consecutive round starts
ROUND_PERIOD_JITTER = 1.0  # ± jitter on round period
MIN_EVENTS = len(PROCESSES)  # 5 init events

# Inter-event delay templates within a round: (mean, ± jitter).
# Causal property `write_enrollment -> (TRUE S write_student)` is enforced
# by message order, so jittering these delays does not affect the verdict.
DELAY_LOCAL_TO_SEND = (0.1, 0.03)  # local op → outbound send
DELAY_NETWORK = (0.8, 0.15)  # send → matching receive
DELAY_RECV_TO_APPLY = (0.2, 0.05)  # receive → local apply
DELAY_APPLY_TO_FORWARD = (0.4, 0.1)  # apply → outbound forward
DELAY_APPLY_TO_WRITE = (0.8, 0.15)  # P5.apply_student → write_enrollment
DELAY_WRITE_TO_SEND = (0.2, 0.05)  # P5.write_enrollment → send_enrollment

# Trailing maintenance ticks (one per process, cycling) used as filler
# when the trace size is not 5 + 17*k. Matches the original Cassandra
# trace's tail. Each tick is a pure local event with no message effect.
MAINTENANCE_TICK_PERIOD = 0.4
MAINTENANCE_PROPS = (
    ("P1", "consistency_check"),
    ("P2", "read_repair"),
    ("P3", "hint_flush"),
    ("P4", "compaction_tick"),
    ("P5", "replica_sync"),
)


# ---------------------------------------------------------------------------
# Generator state and emission helpers
# ---------------------------------------------------------------------------


class GeneratorState:
    """Mutable bookkeeping for clocks, message queues, and the event list."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {p: 0 for p in PROCESSES}
        self.vcs: dict[str, dict[str, int]] = {
            p: {q: 0 for q in PROCESSES} for p in PROCESSES
        }
        self.pending: dict[tuple[str, str], list[dict[str, int]]] = {
            (s, d): [] for s in PROCESSES for d in PROCESSES if s != d
        }
        self.events: list[tuple[str, str, str, str, str, str, str]] = []

    def _vc_str(self, vc: dict[str, int]) -> str:
        return ";".join(f"{p}:{vc[p]}" for p in PROCESSES)

    def _eid(self, proc: str) -> str:
        # Eid follows the existing trace's "<proc-lowercase>_<count>" scheme;
        # called immediately after counts[proc] is incremented.
        return f"{proc.lower()}_{self.counts[proc]}"

    def _record(
        self,
        proc: str,
        timestamp: float,
        props: str,
        etype: str,
        partner: str,
    ) -> None:
        self.events.append(
            (
                self._eid(proc),
                proc,
                self._vc_str(self.vcs[proc]),
                f"{timestamp:.3f}",
                props,
                etype,
                partner,
            )
        )

    def local(self, proc: str, timestamp: float, props: str) -> None:
        self.counts[proc] += 1
        self.vcs[proc][proc] = self.counts[proc]
        self._record(proc, timestamp, props, "local", "")

    def send(
        self, src: str, dst: str, timestamp: float, props: str
    ) -> None:
        self.counts[src] += 1
        self.vcs[src][src] = self.counts[src]
        self.pending[(src, dst)].append(dict(self.vcs[src]))
        self._record(src, timestamp, props, "send", dst)

    def recv(
        self, src: str, dst: str, timestamp: float, props: str
    ) -> None:
        vc_msg = self.pending[(src, dst)].pop(0)
        for q in PROCESSES:
            if q != dst:
                self.vcs[dst][q] = max(self.vcs[dst][q], vc_msg[q])
        self.counts[dst] += 1
        self.vcs[dst][dst] = self.counts[dst]
        self._record(dst, timestamp, props, "receive", src)


# ---------------------------------------------------------------------------
# Per-round event schedule
# ---------------------------------------------------------------------------


def _delay(rng: random.Random, spec: tuple[float, float]) -> float:
    """Return a positive delay drawn around ``spec = (mean, jitter)``."""
    mean, jitter = spec
    return max(0.01, mean + rng.uniform(-jitter, jitter))


def emit_round(state: GeneratorState, round_start: float, rng: random.Random) -> None:
    """Emit the 17-event chain for one round, with jittered inter-event
    delays. The chain ordering and message structure are preserved exactly;
    only timestamps vary."""
    t = round_start
    state.local("P1", t, "write_student")

    t += _delay(rng, DELAY_LOCAL_TO_SEND)
    state.send("P1", "P2", t, "send_student")
    t += _delay(rng, DELAY_NETWORK)
    state.recv("P1", "P2", t, "recv_student")
    t += _delay(rng, DELAY_RECV_TO_APPLY)
    state.local("P2", t, "apply_student")
    t += _delay(rng, DELAY_APPLY_TO_FORWARD)
    state.send("P2", "P3", t, "forward_student")

    t += _delay(rng, DELAY_NETWORK)
    state.recv("P2", "P3", t, "recv_student")
    t += _delay(rng, DELAY_RECV_TO_APPLY)
    state.local("P3", t, "apply_student")
    t += _delay(rng, DELAY_APPLY_TO_FORWARD)
    state.send("P3", "P4", t, "forward_student")

    t += _delay(rng, DELAY_NETWORK)
    state.recv("P3", "P4", t, "recv_student")
    t += _delay(rng, DELAY_RECV_TO_APPLY)
    state.local("P4", t, "apply_student")
    t += _delay(rng, DELAY_APPLY_TO_FORWARD)
    state.send("P4", "P5", t, "forward_student")

    t += _delay(rng, DELAY_NETWORK)
    state.recv("P4", "P5", t, "recv_student")
    t += _delay(rng, DELAY_RECV_TO_APPLY)
    state.local("P5", t, "apply_student")

    t += _delay(rng, DELAY_APPLY_TO_WRITE)
    state.local("P5", t, "write_enrollment")
    t += _delay(rng, DELAY_WRITE_TO_SEND)
    state.send("P5", "P1", t, "send_enrollment")

    t += _delay(rng, DELAY_NETWORK)
    state.recv("P5", "P1", t, "recv_enrollment")
    t += _delay(rng, DELAY_RECV_TO_APPLY)
    state.local("P1", t, "apply_enrollment")


# ---------------------------------------------------------------------------
# Trace assembly
# ---------------------------------------------------------------------------


def emit_init(state: GeneratorState) -> None:
    for p in PROCESSES:
        state.local(p, 0.0, "init")


def emit_maintenance_ticks(
    state: GeneratorState, base_t: float, n: int, rng: random.Random
) -> None:
    """Emit ``n`` trailing maintenance ticks (cycling local events on
    P1..P5) starting from ``base_t``, with each tick period jittered."""
    t = base_t
    for i in range(n):
        proc, label = MAINTENANCE_PROPS[i % len(MAINTENANCE_PROPS)]
        state.local(proc, t, label)
        t += max(
            0.05,
            MAINTENANCE_TICK_PERIOD + rng.uniform(
                -MAINTENANCE_TICK_PERIOD * 0.3, MAINTENANCE_TICK_PERIOD * 0.3
            ),
        )


def emit_trace(target_size: int, seed: int | None) -> GeneratorState:
    if target_size < MIN_EVENTS:
        raise ValueError(
            f"trace size must be at least {MIN_EVENTS} (one init event per process)"
        )

    rng = random.Random(seed)

    state = GeneratorState()
    emit_init(state)

    remaining = target_size - MIN_EVENTS
    full_rounds, partial = divmod(remaining, EVENTS_PER_ROUND)

    round_start = ROUND_START
    for _ in range(full_rounds):
        emit_round(state, round_start, rng)
        period = ROUND_PERIOD + rng.uniform(-ROUND_PERIOD_JITTER, ROUND_PERIOD_JITTER)
        round_start += max(EVENTS_PER_ROUND * 0.1, period)

    if partial:
        emit_maintenance_ticks(state, round_start, partial, rng)

    assert len(state.events) == target_size, (
        f"internal error: emitted {len(state.events)} events, "
        f"expected {target_size}"
    )
    return state


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


HEADER_TEMPLATE = """\
# Cassandra-style distributed DB with reference checks
# total_events: {total}
# process_count: {nproc}
# system_processes: {procs}
# Full rounds: {full_rounds} ({events_per_round} events each, jittered timing)
# Trailing maintenance ticks: {partial}
# Random seed: {seed}
# Round {events_per_round}-event chain: P1.write_student → P2 → P3 → P4 →
#   P5.write_enrollment → P1.apply_enrollment
# Property: write_enrollment -> (TRUE S write_student)
"""


def write_trace(state: GeneratorState, output: Path, seed: int | None) -> None:
    remaining = len(state.events) - MIN_EVENTS
    full_rounds, partial = divmod(remaining, EVENTS_PER_ROUND)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        f.write(
            HEADER_TEMPLATE.format(
                total=len(state.events),
                nproc=len(PROCESSES),
                procs="|".join(PROCESSES),
                full_rounds=full_rounds,
                events_per_round=EVENTS_PER_ROUND,
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
        description="Generate a Cassandra-style trace of the requested length.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Each round contributes {EVENTS_PER_ROUND} events. The trace is\n"
            f"assembled as {MIN_EVENTS} init events + k full rounds + a tail of\n"
            "cycling maintenance ticks (consistency_check, read_repair,\n"
            "hint_flush, compaction_tick, replica_sync) whose length completes\n"
            "the requested total."
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
    full_rounds, partial = divmod(remaining, EVENTS_PER_ROUND)
    print(
        f"wrote {args.events} events to {args.output} "
        f"({MIN_EVENTS} init + {full_rounds} full round × "
        f"{EVENTS_PER_ROUND} + {partial} maintenance tick"
        f"{'s' if partial != 1 else ''})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
