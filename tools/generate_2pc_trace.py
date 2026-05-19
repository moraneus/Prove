#!/usr/bin/env python3
"""
Two-Phase Commit (2PC) trace generator for PROVE experiments.

Generates valid CSV traces with correct vector clocks, FIFO message
ordering, and an ε-sensitive concurrent vote cluster per transaction.

Topology: one coordinator (C) and three participants (P1, P2, P3).
Participants communicate only with the coordinator (asynchronous
FIFO channels); they do not exchange messages with one another.

Per-transaction event layout (33 events):
    1   begin_tx                 (C, local)
    2-4 prepare_P{1,2,3}         (C → Pi, send)
    5-7 recv_prep_P{1,2,3}       (Pi ← C, recv)
    ---- ε-sensitive cluster (VC-concurrent; timing-ordered) ----
    8   vote_yes_P2              (P2, local, label vote_yes_P2)   t+2.50
    9   vote_yes_P1              (P1, local, label vote_yes_P1)   t+2.70
    10  prepared_P1              (P1, local, label prepared_P1)   t+2.75
    11  prepared_P2              (P2, local, label prepared_P2)   t+2.80
    12  vote_yes_P3              (P3, local, label vote_yes_P3)   t+3.00
    13  prepared_P3              (P3, local, label prepared_P3)   t+3.05
    ---- end cluster ----
    14-16 send_yes_P{1,2,3}      (Pi → C, send)
    17-19 recv_yes_P{1,2,3}      (C ← Pi, recv)
    20    decide_commit          (C, local, label committed)
    21-23 commit_cmd_P{1,2,3}    (C → Pi, send)
    24-26 recv_commit_P{1,2,3}   (Pi ← C, recv)
    27-29 local_commit_P{1,2,3}  (Pi, local, label local_commit_Pi)
    30-32 ack_P{1,2,3}           (Pi → C, send)
    33    tx_done                (C, local)

The critical gap is Δt(prepared_P3, prepared_P2) = 0.25s. Under the
property
        TRUE S (prepared_P3 ∧ ⊖ vote_yes_P2)
the verdict flips at ε = 0.25s: SATISFIED when ε ≥ 0.25, VIOLATED
otherwise.

Usage:
    python tools/generate_2pc_trace.py --events 1000 \\
        --output experiments/Safety_2PC/trace_2pc.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROCESSES = ("C", "P1", "P2", "P3")
EVENTS_PER_TX = 33
TX_DURATION = 8.0  # mean seconds of simulated time per transaction
TX_DURATION_JITTER = 0.5  # ± per-tx jitter on the inter-tx gap
MIN_EVENTS = len(PROCESSES)  # 4 init events

# Pre/post-cluster timing jitter (small, kept well below 0.25 so the
# ε-sensitivity threshold is preserved).
PRE_CLUSTER_JITTER = 0.03
POST_CLUSTER_JITTER = 0.05

# Critical cluster offsets (relative to tx start time T). These define the
# ε = 0.25 threshold and MUST stay frozen across runs:
#   Δ(prepared_P3, prepared_P2) = 3.05 - 2.80 = 0.25
CLUSTER_OFFSETS = {
    "vote_yes_P2": 2.50,
    "vote_yes_P1": 2.70,
    "prepared_P1": 2.75,
    "prepared_P2": 2.80,
    "vote_yes_P3": 3.00,
    "prepared_P3": 3.05,
}


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
        for q in PROCESSES:
            if q != dst:
                self.vcs[dst][q] = max(self.vcs[dst][q], vc_msg[q])
        self.counts[dst] += 1
        self.vcs[dst][dst] = self.counts[dst]
        self._record(eid, dst, timestamp, props, "receive", src)


# ---------------------------------------------------------------------------
# Per-transaction event schedule
# ---------------------------------------------------------------------------


def _jit_pre(rng: random.Random, base: float) -> float:
    """Jitter a pre-cluster offset. Bounded so the result stays well below
    the cluster start at 2.50."""
    return base + rng.uniform(-PRE_CLUSTER_JITTER, PRE_CLUSTER_JITTER)


def _jit_post(rng: random.Random, base: float) -> float:
    """Jitter a post-cluster offset. Bounded so the result stays well above
    the cluster end at 3.05."""
    return base + rng.uniform(-POST_CLUSTER_JITTER, POST_CLUSTER_JITTER)


def tx_step_functions(
    state: GeneratorState, N: int, T: float, rng: random.Random
) -> list:
    """Return the 33 emission closures for transaction N (1-indexed),
    starting at simulated time ``T``.

    Cluster offsets (events 8-13) are FROZEN: their absolute timestamps
    relative to T must be exactly the values in CLUSTER_OFFSETS so that
    the ε-sensitivity threshold of 0.25 is preserved across runs.
    Pre-cluster (events 1-7) and post-cluster (events 14-33) offsets are
    jittered slightly.
    """
    sfx = f"_{N:02d}"

    # Pre-cluster: jittered, but each base is far enough from 2.50 that
    # PRE_CLUSTER_JITTER cannot push them past it.
    pre = {
        "begin": _jit_pre(rng, 1.0),
        "prepare_P1": _jit_pre(rng, 1.1),
        "prepare_P2": _jit_pre(rng, 1.2),
        "prepare_P3": _jit_pre(rng, 1.3),
        "recv_prep_P1": _jit_pre(rng, 2.0),
        "recv_prep_P2": _jit_pre(rng, 2.1),
        "recv_prep_P3": _jit_pre(rng, 2.2),
    }
    # Post-cluster: each base is far enough from 3.05 that
    # POST_CLUSTER_JITTER cannot push them earlier.
    post = {
        "send_yes_P1": _jit_post(rng, 3.3),
        "send_yes_P2": _jit_post(rng, 3.4),
        "send_yes_P3": _jit_post(rng, 3.5),
        "recv_yes_P1": _jit_post(rng, 4.0),
        "recv_yes_P2": _jit_post(rng, 4.1),
        "recv_yes_P3": _jit_post(rng, 4.2),
        "decide_commit": _jit_post(rng, 4.5),
        "commit_cmd_P1": _jit_post(rng, 4.7),
        "commit_cmd_P2": _jit_post(rng, 4.8),
        "commit_cmd_P3": _jit_post(rng, 4.9),
        "recv_commit_P1": _jit_post(rng, 5.5),
        "recv_commit_P2": _jit_post(rng, 5.6),
        "recv_commit_P3": _jit_post(rng, 5.7),
        "local_commit_P1": _jit_post(rng, 6.0),
        "local_commit_P2": _jit_post(rng, 6.1),
        "local_commit_P3": _jit_post(rng, 6.2),
        "ack_P1": _jit_post(rng, 6.5),
        "ack_P2": _jit_post(rng, 6.6),
        "ack_P3": _jit_post(rng, 6.7),
        "tx_done": _jit_post(rng, 7.5),
    }

    return [
        # 1. begin
        lambda: state.local(f"begin_tx{sfx}", "C", T + pre["begin"]),
        # 2-4. prepare fan-out
        lambda: state.send(f"prepare_P1{sfx}", "C", "P1", T + pre["prepare_P1"]),
        lambda: state.send(f"prepare_P2{sfx}", "C", "P2", T + pre["prepare_P2"]),
        lambda: state.send(f"prepare_P3{sfx}", "C", "P3", T + pre["prepare_P3"]),
        # 5-7. prepare receipts
        lambda: state.recv(f"recv_prep_P1{sfx}", "C", "P1", T + pre["recv_prep_P1"]),
        lambda: state.recv(f"recv_prep_P2{sfx}", "C", "P2", T + pre["recv_prep_P2"]),
        lambda: state.recv(f"recv_prep_P3{sfx}", "C", "P3", T + pre["recv_prep_P3"]),
        # 8-13. ε-sensitive cluster — FROZEN
        lambda: state.local(
            f"vote_yes_P2{sfx}", "P2", T + CLUSTER_OFFSETS["vote_yes_P2"], "vote_yes_P2"
        ),
        lambda: state.local(
            f"vote_yes_P1{sfx}", "P1", T + CLUSTER_OFFSETS["vote_yes_P1"], "vote_yes_P1"
        ),
        lambda: state.local(
            f"prepared_P1{sfx}", "P1", T + CLUSTER_OFFSETS["prepared_P1"], "prepared_P1"
        ),
        lambda: state.local(
            f"prepared_P2{sfx}", "P2", T + CLUSTER_OFFSETS["prepared_P2"], "prepared_P2"
        ),
        lambda: state.local(
            f"vote_yes_P3{sfx}", "P3", T + CLUSTER_OFFSETS["vote_yes_P3"], "vote_yes_P3"
        ),
        lambda: state.local(
            f"prepared_P3{sfx}", "P3", T + CLUSTER_OFFSETS["prepared_P3"], "prepared_P3"
        ),
        # 14-16. votes returned
        lambda: state.send(f"send_yes_P1{sfx}", "P1", "C", T + post["send_yes_P1"]),
        lambda: state.send(f"send_yes_P2{sfx}", "P2", "C", T + post["send_yes_P2"]),
        lambda: state.send(f"send_yes_P3{sfx}", "P3", "C", T + post["send_yes_P3"]),
        # 17-19. vote receipts
        lambda: state.recv(f"recv_yes_P1{sfx}", "P1", "C", T + post["recv_yes_P1"]),
        lambda: state.recv(f"recv_yes_P2{sfx}", "P2", "C", T + post["recv_yes_P2"]),
        lambda: state.recv(f"recv_yes_P3{sfx}", "P3", "C", T + post["recv_yes_P3"]),
        # 20. decide
        lambda: state.local(
            f"decide_commit{sfx}", "C", T + post["decide_commit"], "committed"
        ),
        # 21-23. commit fan-out
        lambda: state.send(
            f"commit_cmd_P1{sfx}", "C", "P1", T + post["commit_cmd_P1"]
        ),
        lambda: state.send(
            f"commit_cmd_P2{sfx}", "C", "P2", T + post["commit_cmd_P2"]
        ),
        lambda: state.send(
            f"commit_cmd_P3{sfx}", "C", "P3", T + post["commit_cmd_P3"]
        ),
        # 24-26. commit receipts
        lambda: state.recv(
            f"recv_commit_P1{sfx}", "C", "P1", T + post["recv_commit_P1"]
        ),
        lambda: state.recv(
            f"recv_commit_P2{sfx}", "C", "P2", T + post["recv_commit_P2"]
        ),
        lambda: state.recv(
            f"recv_commit_P3{sfx}", "C", "P3", T + post["recv_commit_P3"]
        ),
        # 27-29. local commits
        lambda: state.local(
            f"local_commit_P1{sfx}", "P1", T + post["local_commit_P1"], "local_commit_P1"
        ),
        lambda: state.local(
            f"local_commit_P2{sfx}", "P2", T + post["local_commit_P2"], "local_commit_P2"
        ),
        lambda: state.local(
            f"local_commit_P3{sfx}", "P3", T + post["local_commit_P3"], "local_commit_P3"
        ),
        # 30-32. acks
        lambda: state.send(f"ack_P1{sfx}", "P1", "C", T + post["ack_P1"]),
        lambda: state.send(f"ack_P2{sfx}", "P2", "C", T + post["ack_P2"]),
        lambda: state.send(f"ack_P3{sfx}", "P3", "C", T + post["ack_P3"]),
        # 33. done
        lambda: state.local(f"tx_done{sfx}", "C", T + post["tx_done"]),
    ]


# ---------------------------------------------------------------------------
# Trace assembly
# ---------------------------------------------------------------------------


def emit_init(state: GeneratorState) -> None:
    for p in PROCESSES:
        state.local(f"iota_{p}", p, 0.0, "init")


def emit_trace(target_size: int, seed: int | None) -> GeneratorState:
    if target_size < MIN_EVENTS:
        raise ValueError(
            f"trace size must be at least {MIN_EVENTS} (one init event per process)"
        )

    rng = random.Random(seed)

    state = GeneratorState()
    emit_init(state)

    remaining = target_size - MIN_EVENTS
    full_txs, partial = divmod(remaining, EVENTS_PER_TX)

    T = 0.0
    for N in range(1, full_txs + 1):
        for step in tx_step_functions(state, N, T, rng):
            step()
        T += TX_DURATION + rng.uniform(-TX_DURATION_JITTER, TX_DURATION_JITTER)

    if partial:
        for step in tx_step_functions(state, full_txs + 1, T, rng)[:partial]:
            step()

    assert len(state.events) == target_size, (
        f"internal error: emitted {len(state.events)} events, "
        f"expected {target_size}"
    )
    return state


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


HEADER_TEMPLATE = """\
# Two-Phase Commit: 1 Coordinator + 3 Participants
# total_events: {total}
# process_count: {nproc}
# system_processes: {procs}
# Full transactions: {full_txs} ({events_per_tx} events each, jittered timing)
# Partial transaction events: {partial}
# Random seed: {seed}
# Per-transaction ε-sensitive cluster (VC-concurrent; timing-ordered, FROZEN):
#   vote_yes_P2 t+2.50 | vote_yes_P1 t+2.70 | prepared_P1 t+2.75
#   prepared_P2 t+2.80 | vote_yes_P3 t+3.00 | prepared_P3 t+3.05
# Critical gap: Δ(prepared_P3, prepared_P2) = 0.25
# Property: TRUE S (prepared_P3 & @vote_yes_P2)
#   SAT  when ε >= 0.25 (prepared_P3 and prepared_P2 commutable)
#   VIOL when ε <  0.25 (prepared_P2 forced before prepared_P3)
"""


def write_trace(state: GeneratorState, output: Path, seed: int | None) -> None:
    remaining = len(state.events) - MIN_EVENTS
    full_txs, partial = divmod(remaining, EVENTS_PER_TX)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        f.write(
            HEADER_TEMPLATE.format(
                total=len(state.events),
                nproc=len(PROCESSES),
                procs="|".join(PROCESSES),
                full_txs=full_txs,
                events_per_tx=EVENTS_PER_TX,
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
        description="Generate a 2PC trace of the requested length.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Each transaction contributes 33 events. The trace is assembled\n"
            "as 4 init events + k full transactions + a partial transaction\n"
            "prefix whose length completes the requested total."
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
        help=(
            "random seed for jittered timing (default: time-based). "
            "The ε-sensitive cluster is frozen across all seeds."
        ),
    )
    args = parser.parse_args(argv)

    try:
        state = emit_trace(args.events, args.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    write_trace(state, args.output, args.seed)

    remaining = args.events - MIN_EVENTS
    full_txs, partial = divmod(remaining, EVENTS_PER_TX)
    print(
        f"wrote {args.events} events to {args.output} "
        f"({MIN_EVENTS} init + {full_txs} full tx × {EVENTS_PER_TX} "
        f"+ {partial} partial)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
