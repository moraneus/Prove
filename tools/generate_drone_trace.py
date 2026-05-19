#!/usr/bin/env python3
"""
Drone swarm trace generator for PROVE scalability experiments.

Generates valid CSV traces with correct vector clocks, FIFO message
ordering, and realistic mission-phase propositions for N drones.

Includes a "fragile" epsilon scenario: two VC-concurrent events on
different drones are placed with a configurable timestamp gap. When
epsilon >= gap, they remain concurrent (property SATISFIED); when
epsilon < gap, time forces ordering (property VIOLATED).

Usage:
    python tools/generate_drone_trace.py \\
        --events 10000 --drones 3 --message-rate 0.3 \\
        --fragile-delta 2.0 --seed 42 \\
        --output experiments/drone_swarm/traces/trace_10k.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Mission phases that drones cycle through
MISSION_PHASES = [
    "idle",
    "scanning",
    "target_found",
    "position_measured",
    "position_sent",
    "processing",
    "relative_computed",
    "formation_hold",
]

PHASE_CYCLE_LENGTH = 10  # events per drone before cycling phases


@dataclass
class DroneState:
    """Mutable state for a single drone during trace generation."""

    name: str
    vc_own: int = 0  # own VC component (will be 1 after init)
    timestamp: float = 0.0
    phase_index: int = 0
    events_in_phase: int = 0
    event_count: int = 0


@dataclass
class GeneratorState:
    """Global generator state."""

    drones: list[DroneState]
    drone_names: list[str]
    # VC: full vector clock per drone (drone_name -> {drone_name -> int})
    vc: dict[str, dict[str, int]]
    # FIFO message queues: (sender, receiver) -> list of (sender_vc, send_timestamp)
    msg_queues: dict[tuple[str, str], list[tuple[dict[str, int], float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Pending receives: (receiver_drone_name) -> list of sender names
    pending_receives: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    total_events: int = 0
    fragile_injected: bool = False


def create_initial_events(
    state: GeneratorState,
) -> list[dict[str, str]]:
    """Create initial events (one per drone), all at t=0.0."""
    rows = []
    for drone in state.drones:
        drone.vc_own = 1
        state.vc[drone.name][drone.name] = 1
        drone.event_count = 1

        vc_str = format_vc(state.vc[drone.name], state.drone_names)
        rows.append(
            {
                "eid": f"init_{drone.name}",
                "processes": drone.name,
                "vc": vc_str,
                "timestamp": "0.0",
                "props": "idle",
                "event_type": "local",
                "msg_partner": "",
            }
        )
    state.total_events = len(state.drones)
    return rows


def format_vc(vc: dict[str, int], drone_names: list[str]) -> str:
    """Format vector clock as 'Drone1:2;Drone2:1;Drone3:0'."""
    return ";".join(f"{name}:{vc.get(name, 0)}" for name in drone_names)


def increment_vc(state: GeneratorState, drone_name: str) -> None:
    """Increment a drone's own VC component."""
    state.vc[drone_name][drone_name] += 1


def merge_vc(state: GeneratorState, receiver: str, sender_vc: dict[str, int]) -> None:
    """Merge sender's VC into receiver's VC (Fidge-Mattern rule)."""
    # Increment own component
    state.vc[receiver][receiver] += 1
    # Merge: max of each component (except own, already incremented)
    for k, v in sender_vc.items():
        if k != receiver:
            state.vc[receiver][k] = max(state.vc[receiver].get(k, 0), v)


def get_current_prop(drone: DroneState) -> str:
    """Get the current mission phase proposition for a drone."""
    return MISSION_PHASES[drone.phase_index % len(MISSION_PHASES)]


def advance_phase(drone: DroneState) -> None:
    """Advance drone's mission phase if enough events have passed."""
    drone.events_in_phase += 1
    if drone.events_in_phase >= PHASE_CYCLE_LENGTH:
        drone.events_in_phase = 0
        drone.phase_index += 1


def generate_local_event(
    state: GeneratorState,
    drone: DroneState,
    rng: random.Random,
    extra_props: str | None = None,
) -> dict[str, str]:
    """Generate a local event for a drone."""
    increment_vc(state, drone.name)
    drone.vc_own = state.vc[drone.name][drone.name]
    drone.timestamp += rng.uniform(0.05, 0.3)
    drone.event_count += 1
    state.total_events += 1

    prop = extra_props if extra_props else get_current_prop(drone)
    advance_phase(drone)

    eid = f"{drone.name}_e{drone.event_count}"
    return {
        "eid": eid,
        "processes": drone.name,
        "vc": format_vc(state.vc[drone.name], state.drone_names),
        "timestamp": f"{drone.timestamp:.4f}",
        "props": prop,
        "event_type": "local",
        "msg_partner": "",
    }


def generate_send_event(
    state: GeneratorState,
    sender: DroneState,
    receiver_name: str,
    rng: random.Random,
    extra_props: str | None = None,
) -> dict[str, str]:
    """Generate a send event from sender to receiver."""
    increment_vc(state, sender.name)
    sender.vc_own = state.vc[sender.name][sender.name]
    sender.timestamp += rng.uniform(0.05, 0.3)
    sender.event_count += 1
    state.total_events += 1

    # Snapshot the sender's VC and timestamp for the message
    vc_snapshot = dict(state.vc[sender.name])
    state.msg_queues[(sender.name, receiver_name)].append((vc_snapshot, sender.timestamp))

    # Track pending receives for the receiver
    state.pending_receives[receiver_name].append(sender.name)

    prop = extra_props if extra_props else get_current_prop(sender)
    advance_phase(sender)

    eid = f"{sender.name}_e{sender.event_count}"
    return {
        "eid": eid,
        "processes": sender.name,
        "vc": format_vc(state.vc[sender.name], state.drone_names),
        "timestamp": f"{sender.timestamp:.4f}",
        "props": prop,
        "event_type": "send",
        "msg_partner": receiver_name,
    }


def generate_receive_event(
    state: GeneratorState,
    receiver: DroneState,
    sender_name: str,
    rng: random.Random,
    extra_props: str | None = None,
) -> dict[str, str]:
    """Generate a receive event on receiver from sender."""
    # Pop the oldest message from this sender (FIFO)
    queue = state.msg_queues[(sender_name, receiver.name)]
    if not queue:
        raise ValueError(f"No pending message from {sender_name} to {receiver.name}")
    sender_vc, send_timestamp = queue.pop(0)

    # Also remove from pending_receives
    pending = state.pending_receives[receiver.name]
    for i, s in enumerate(pending):
        if s == sender_name:
            pending.pop(i)
            break

    # Merge VCs
    merge_vc(state, receiver.name, sender_vc)
    receiver.vc_own = state.vc[receiver.name][receiver.name]
    # Ensure receive timestamp > send timestamp (causal consistency)
    receiver.timestamp = max(receiver.timestamp, send_timestamp) + rng.uniform(0.05, 0.3)
    receiver.event_count += 1
    state.total_events += 1

    prop = extra_props if extra_props else get_current_prop(receiver)
    advance_phase(receiver)

    eid = f"{receiver.name}_e{receiver.event_count}"
    return {
        "eid": eid,
        "processes": receiver.name,
        "vc": format_vc(state.vc[receiver.name], state.drone_names),
        "timestamp": f"{receiver.timestamp:.4f}",
        "props": prop,
        "event_type": "receive",
        "msg_partner": sender_name,
    }


def can_receive(state: GeneratorState, receiver_name: str) -> list[str]:
    """Return list of senders that have pending messages for receiver."""
    senders = []
    for (s, r), queue in state.msg_queues.items():
        if r == receiver_name and queue:
            senders.append(s)
    return senders


def inject_fragile_scenario(
    state: GeneratorState,
    fragile_delta: float,
) -> list[dict[str, str]]:
    """
    Inject the fragile epsilon scenario as the LAST events on Drone1 and Drone2.

    Creates two VC-concurrent events:
    - E_late on Drone2: props="late_beacon", timestamp=T (earlier)
    - E_early on Drone1: props="early_beacon", timestamp=T+delta (later)

    These are the LAST events on their respective processes, so they
    remain in the final frontier's global state.

    Property: late_beacon -> @early_beacon
    ("if late_beacon is true, then early_beacon was true at the previous step")

    - eps >= delta: events concurrent, linearization exists where E_early
      is processed before E_late. At the step where E_late is processed,
      early_beacon is already in the global state (Drone1's maximal = E_early).
      So @early_beacon = TRUE. Property: TRUE -> TRUE = TRUE. SAT.

    - eps < delta: t(E_early) - t(E_late) = delta > eps forces E_late < E_early.
      E_late is processed first. At that step, Drone1's maximal is its
      second-to-last event (not early_beacon). @early_beacon = FALSE.
      Later E_early is processed last; Drone2's maximal is E_late (still).
      At this step @early_beacon = FALSE (prev step's Drone1 maximal was
      the second-to-last). Property: TRUE -> FALSE = FALSE. VIOLATED.
    """
    rows = []

    d1 = state.drones[0]
    d2 = state.drones[1]

    # Synchronize timestamps to a common base
    base_t = max(d1.timestamp, d2.timestamp) + 1.0

    # E_late on Drone2 (earlier timestamp)
    d2.timestamp = base_t
    increment_vc(state, d2.name)
    d2.vc_own = state.vc[d2.name][d2.name]
    d2.event_count += 1
    state.total_events += 1
    rows.append(
        {
            "eid": f"{d2.name}_fragile",
            "processes": d2.name,
            "vc": format_vc(state.vc[d2.name], state.drone_names),
            "timestamp": f"{d2.timestamp:.4f}",
            "props": "late_beacon",
            "event_type": "local",
            "msg_partner": "",
        }
    )

    # E_early on Drone1 (later timestamp by delta)
    d1.timestamp = base_t + fragile_delta
    increment_vc(state, d1.name)
    d1.vc_own = state.vc[d1.name][d1.name]
    d1.event_count += 1
    state.total_events += 1
    rows.append(
        {
            "eid": f"{d1.name}_fragile",
            "processes": d1.name,
            "vc": format_vc(state.vc[d1.name], state.drone_names),
            "timestamp": f"{d1.timestamp:.4f}",
            "props": "early_beacon",
            "event_type": "local",
            "msg_partner": "",
        }
    )

    state.fragile_injected = True
    return rows


def generate_trace(
    num_events: int,
    num_drones: int,
    message_rate: float,
    fragile_delta: float,
    seed: int,
) -> str:
    """Generate a complete drone swarm trace as CSV string."""
    rng = random.Random(seed)

    drone_names = [f"Drone{i+1}" for i in range(num_drones)]

    # Initialize state
    state = GeneratorState(
        drones=[DroneState(name=name) for name in drone_names],
        drone_names=drone_names,
        vc={name: {n: 0 for n in drone_names} for name in drone_names},
    )

    all_rows: list[dict[str, str]] = []

    # Initial events
    all_rows.extend(create_initial_events(state))

    # Reserve 2 slots for fragile events at the end
    target_normal_events = num_events - 2 if num_drones >= 2 else num_events

    # Generate normal events (all drones participate)
    while state.total_events < target_normal_events:
        # Pick a random drone
        drone = rng.choice(state.drones)

        # Check if this drone has pending receives to process
        available_senders = can_receive(state, drone.name)

        # Decide event type
        roll = rng.random()

        if available_senders and roll < 0.4:
            # Process a pending receive (prioritize clearing the queue)
            sender_name = rng.choice(available_senders)
            row = generate_receive_event(state, drone, sender_name, rng)
            all_rows.append(row)

        elif roll < message_rate and num_drones > 1:
            # Send a message to another drone
            other_drones = [d for d in drone_names if d != drone.name]
            receiver_name = rng.choice(other_drones)
            row = generate_send_event(state, drone, receiver_name, rng)
            all_rows.append(row)

        else:
            # Local event
            row = generate_local_event(state, drone, rng)
            all_rows.append(row)

    # Drain remaining pending receives before fragile events
    for drone in state.drones:
        while can_receive(state, drone.name):
            senders = can_receive(state, drone.name)
            for sender_name in senders:
                row = generate_receive_event(state, drone, sender_name, rng)
                all_rows.append(row)

    # Inject fragile events as the LAST events on Drone1 and Drone2
    if num_drones >= 2:
        all_rows.extend(inject_fragile_scenario(state, fragile_delta))

    # Build CSV output
    output = io.StringIO()
    output.write(f"# Drone swarm trace: {num_drones} drones, {len(all_rows)} events\n")
    output.write(f"# system_processes: {'|'.join(drone_names)}\n")

    writer = csv.DictWriter(
        output,
        fieldnames=["eid", "processes", "vc", "timestamp", "props", "event_type", "msg_partner"],
    )
    writer.writeheader()
    writer.writerows(all_rows)

    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate drone swarm traces for PROVE experiments"
    )
    parser.add_argument(
        "--events",
        type=int,
        default=1000,
        help="Target number of events (default: 1000)",
    )
    parser.add_argument(
        "--drones",
        type=int,
        default=3,
        help="Number of drones (default: 3)",
    )
    parser.add_argument(
        "--message-rate",
        type=float,
        default=0.3,
        help="Fraction of events that are sends (default: 0.3)",
    )
    parser.add_argument(
        "--fragile-delta",
        type=float,
        default=2.0,
        help="Timestamp gap for fragile scenario (default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    csv_content = generate_trace(
        num_events=args.events,
        num_drones=args.drones,
        message_rate=args.message_rate,
        fragile_delta=args.fragile_delta,
        seed=args.seed,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(csv_content)
        print(f"Generated {args.output} ({args.events} target events)", file=sys.stderr)
    else:
        sys.stdout.write(csv_content)


if __name__ == "__main__":
    main()
