#!/usr/bin/env python3
"""
Drone relative-velocity trace generator for PROVE experiments.

Generates valid CSV traces matching the structure used by
``experiments/Safety_Drone``: one orchestrator drone (Drone1) that
sequentially exchanges velocity / relative-velocity messages with each
peer (Drone2 .. DroneK) in order. The number of drones K is
configurable.

Per-round event layout for K drones (5K - 4 events per round):

    1                       Drone1.velocity_measured (local)         T
    For each peer i = 2..K:
        Drone1.velocity_sent        (Drone1 → Drone_i, send)         T + 0.2 + p·4.0
        Drone_i.processing          (Drone_i ← Drone1, receive)      T + 1.0 + p·4.0
        Drone_i.relative_computed   (local)                          T + 1.5 + p·4.0
        Drone_i.relative_sent       (Drone_i → Drone1, send)         T + 2.0 + p·4.0
        Drone1.relative_confirmed   (Drone1 ← Drone_i, receive)      T + 3.0 + p·4.0
    where p = i - 2 (0-indexed peer position).

Rounds are spaced 4K - 3 seconds apart; round 1 begins at T = 1.0.

The reference-check property
        relative_confirmed → (TRUE S velocity_sent)
is causally satisfied (every relative_confirmed on Drone1 is the
endpoint of a chain rooted at the matching velocity_sent), so the
trace verdict is SATISFIED for any ε.

Usage:
    python tools/generate_safety_drone_trace.py \\
        --events 1000 --drones 10 \\
        --output experiments/Safety_Drone/trace_drone.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Inter-event delays inside a peer block (cumulative from previous event):
# Drone1.measure → Drone1.send → peer.recv → peer.compute → peer.send_back
# → Drone1.recv_back → next peer's Drone1.send. Each is (mean, ± jitter).
# Property `relative_confirmed -> (TRUE S velocity_sent)` is causally enforced,
# so jittering these delays does not affect the verdict.
# ---------------------------------------------------------------------------
DELAY_MEASURE_TO_SEND = (0.2, 0.05)  # Drone1.measure → Drone1.send (first peer)
DELAY_SEND_TO_RECV = (0.8, 0.15)  # Drone1.send → peer.recv
DELAY_RECV_TO_COMPUTE = (0.5, 0.1)  # peer.recv → peer.compute
DELAY_COMPUTE_TO_SEND = (0.5, 0.1)  # peer.compute → peer.send_back
DELAY_SEND_TO_CONFIRM = (1.0, 0.15)  # peer.send_back → Drone1.recv
DELAY_CONFIRM_TO_NEXT = (1.2, 0.2)  # Drone1.recv → Drone1.send (next peer)

ROUND_START = 1.0
INTER_ROUND_GAP = 2.0  # mean quiet time between rounds (jittered ±25%)
INTER_ROUND_GAP_JITTER = 0.5

EVENTS_PER_PEER = 5
EVENTS_PER_ROUND_BASE = 1  # the leading velocity_measured on Drone1


# ---------------------------------------------------------------------------
# Generator state and emission helpers
# ---------------------------------------------------------------------------


class GeneratorState:
    """Mutable bookkeeping for clocks, message queues, and the event list."""

    def __init__(self, processes: tuple[str, ...]) -> None:
        self.processes = processes
        self.counts: dict[str, int] = {p: 0 for p in processes}
        self.vcs: dict[str, dict[str, int]] = {
            p: {q: 0 for q in processes} for p in processes
        }
        self.pending: dict[tuple[str, str], list[dict[str, int]]] = {
            (s, d): [] for s in processes for d in processes if s != d
        }
        self.events: list[tuple[str, str, str, str, str, str, str]] = []

    def _vc_str(self, vc: dict[str, int]) -> str:
        return ";".join(f"{p}:{vc[p]}" for p in self.processes)

    def _record(
        self,
        eid: str,
        proc: str,
        timestamp: float,
        props: str,
        etype: str,
        partner: str,
    ) -> None:
        self.events.append(
            (
                eid,
                proc,
                self._vc_str(self.vcs[proc]),
                f"{timestamp:.3f}",
                props,
                etype,
                partner,
            )
        )

    def local(self, eid: str, proc: str, timestamp: float, props: str = "") -> None:
        self.counts[proc] += 1
        self.vcs[proc][proc] = self.counts[proc]
        self._record(eid, proc, timestamp, props, "local", "")

    def send(
        self, eid: str, src: str, dst: str, timestamp: float, props: str = ""
    ) -> None:
        self.counts[src] += 1
        self.vcs[src][src] = self.counts[src]
        self.pending[(src, dst)].append(dict(self.vcs[src]))
        self._record(eid, src, timestamp, props, "send", dst)

    def recv(
        self, eid: str, src: str, dst: str, timestamp: float, props: str = ""
    ) -> None:
        vc_msg = self.pending[(src, dst)].pop(0)
        for q in self.processes:
            if q != dst:
                self.vcs[dst][q] = max(self.vcs[dst][q], vc_msg[q])
        self.counts[dst] += 1
        self.vcs[dst][dst] = self.counts[dst]
        self._record(eid, dst, timestamp, props, "receive", src)


# ---------------------------------------------------------------------------
# Per-round event schedule
# ---------------------------------------------------------------------------


def _delay(rng: random.Random, spec: tuple[float, float]) -> float:
    """Return a positive delay drawn around ``spec = (mean, jitter)``."""
    mean, jitter = spec
    return max(0.01, mean + rng.uniform(-jitter, jitter))


def emit_round(
    state: GeneratorState,
    K: int,
    round_idx: int,
    round_start: float,
    rng: random.Random,
    peers_to_emit: int | None = None,
) -> None:
    """Emit one round (or a prefix of it). With ``peers_to_emit`` events,
    emit only that many events from the round.

    A full round produces ``5K - 4`` events: 1 measure + 5*(K-1) peer-block
    events. The chain order and message structure are preserved exactly;
    only timestamps are jittered.
    """
    rr = f"{round_idx:02d}"
    full_round_events = 1 + 5 * (K - 1)
    target = full_round_events if peers_to_emit is None else peers_to_emit
    emitted = 0

    t = round_start
    state.local(f"d1_measure_{rr}", "Drone1", t, "velocity_measured")
    emitted += 1
    if emitted >= target:
        return

    for p, peer_idx in enumerate(range(2, K + 1)):
        peer = f"Drone{peer_idx}"

        # Drone1 → peer: velocity_sent
        if p == 0:
            t += _delay(rng, DELAY_MEASURE_TO_SEND)
        else:
            t += _delay(rng, DELAY_CONFIRM_TO_NEXT)
        state.send(f"d1_send_d{peer_idx}_{rr}", "Drone1", peer, t, "velocity_sent")
        emitted += 1
        if emitted >= target:
            return

        # peer.recv: processing
        t += _delay(rng, DELAY_SEND_TO_RECV)
        state.recv(f"d{peer_idx}_recv_d1_{rr}", "Drone1", peer, t, "processing")
        emitted += 1
        if emitted >= target:
            return

        # peer: relative_computed
        t += _delay(rng, DELAY_RECV_TO_COMPUTE)
        state.local(f"d{peer_idx}_compute_{rr}", peer, t, "relative_computed")
        emitted += 1
        if emitted >= target:
            return

        # peer → Drone1: relative_sent
        t += _delay(rng, DELAY_COMPUTE_TO_SEND)
        state.send(f"d{peer_idx}_send_d1_{rr}", peer, "Drone1", t, "relative_sent")
        emitted += 1
        if emitted >= target:
            return

        # Drone1 ← peer: relative_confirmed
        t += _delay(rng, DELAY_SEND_TO_CONFIRM)
        state.recv(
            f"d1_recv_d{peer_idx}_{rr}", peer, "Drone1", t, "relative_confirmed"
        )
        emitted += 1
        if emitted >= target:
            return


# ---------------------------------------------------------------------------
# Trace assembly
# ---------------------------------------------------------------------------


def emit_init(state: GeneratorState) -> None:
    for proc in state.processes:
        idx = int(proc[len("Drone") :])
        state.local(f"d{idx}_init", proc, 0.0, "stable")


def emit_trace(target_size: int, drones: int, seed: int | None) -> GeneratorState:
    if drones < 2:
        raise ValueError("drones must be at least 2")
    if target_size < drones:
        raise ValueError(
            f"trace size must be at least {drones} (one init event per drone)"
        )

    rng = random.Random(seed)

    processes = tuple(f"Drone{i}" for i in range(1, drones + 1))
    state = GeneratorState(processes)
    emit_init(state)

    events_per_round = EVENTS_PER_ROUND_BASE + EVENTS_PER_PEER * (drones - 1)

    remaining = target_size - drones
    full_rounds, partial = divmod(remaining, events_per_round)

    round_start = ROUND_START
    for r in range(1, full_rounds + 1):
        emit_round(state, drones, r, round_start, rng)
        gap = INTER_ROUND_GAP + rng.uniform(
            -INTER_ROUND_GAP_JITTER, INTER_ROUND_GAP_JITTER
        )
        # Round_start advances to (last event in this round) + gap; the
        # last event timestamp lives in state.events[-1][3].
        last_t = float(state.events[-1][3])
        round_start = last_t + max(0.5, gap)

    if partial:
        emit_round(state, drones, full_rounds + 1, round_start, rng, peers_to_emit=partial)

    assert len(state.events) == target_size, (
        f"internal error: emitted {len(state.events)} events, "
        f"expected {target_size}"
    )
    return state


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


HEADER_TEMPLATE = """\
# Generated drone relative-velocity trace
# total_events: {total}
# drone_count: {drones}
# system_processes: {procs}
# Full rounds: {full_rounds} ({events_per_round} events each, jittered timing)
# Partial round events: {partial}
# Random seed: {seed}
# Round structure: Drone1.velocity_measured → for each peer Drone_i (i=2..K):
#   Drone1→Drone_i velocity_sent → Drone_i processing → relative_computed →
#   Drone_i→Drone1 relative_sent → Drone1 relative_confirmed
# Property: relative_confirmed -> (TRUE S velocity_sent)
"""


def write_trace(
    state: GeneratorState, drones: int, output: Path, seed: int | None
) -> None:
    events_per_round = EVENTS_PER_ROUND_BASE + EVENTS_PER_PEER * (drones - 1)
    remaining = len(state.events) - drones
    full_rounds, partial = divmod(remaining, events_per_round)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        f.write(
            HEADER_TEMPLATE.format(
                total=len(state.events),
                drones=drones,
                procs="|".join(state.processes),
                full_rounds=full_rounds,
                events_per_round=events_per_round,
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
        description=(
            "Generate a Drone relative-velocity trace of the requested length, "
            "for a configurable number of drones."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "For K drones, each round contributes 5K - 4 events:\n"
            "  1 velocity_measured on Drone1 +\n"
            "  K-1 peer blocks of 5 events each (send, recv, compute,\n"
            "  send-back, recv-back).\n\n"
            "The trace is assembled as K init events + r full rounds + a\n"
            "partial round prefix whose length completes the requested total."
        ),
    )
    parser.add_argument(
        "--events",
        "-n",
        type=int,
        required=True,
        help="target number of events (>= number of drones)",
    )
    parser.add_argument(
        "--drones",
        "-k",
        type=int,
        default=10,
        help="number of drones (>= 2, default: 10)",
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
        state = emit_trace(args.events, args.drones, args.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    write_trace(state, args.drones, args.output, args.seed)

    events_per_round = EVENTS_PER_ROUND_BASE + EVENTS_PER_PEER * (args.drones - 1)
    remaining = args.events - args.drones
    full_rounds, partial = divmod(remaining, events_per_round)
    print(
        f"wrote {args.events} events to {args.output} "
        f"({args.drones} init + {full_rounds} full round × "
        f"{events_per_round} + {partial} partial)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
