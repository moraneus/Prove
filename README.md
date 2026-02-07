# PROVE -- Partial oRder Verification Engine

Runtime verification of Existential Past Linear Temporal Logic (EPLTL) properties over partial order executions of distributed systems.

## Overview

PROVE implements a sliding window algorithm for checking whether **at least one valid linearization** of a partial order execution satisfies a given past-time temporal property. It combines:

- **Fidge-Mattern vector clocks** for causal ordering
- **Bounded clock skew (epsilon)** for additional temporal ordering
- **Sliding window graph** for efficient state space management
- **Summary-based evaluation** for EPLTL formula checking

Based on the paper: *"Runtime Verification of Linear Temporal Properties over Partial Order Executions"* by Doron Peled et al.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/moraneus/Prove.git
cd Prove
pip install -e ".[dev]"
```

## Quick Start

```bash
# Check a property against a trace
prove -p examples/properties/safety.prop -t examples/traces/simple.csv

# With epsilon (clock skew bound)
prove -p examples/properties/response.prop -t examples/traces/message_passing.csv -e 2.0

# Verbose output with ASCII visualization
prove -p examples/properties/safety.prop -t examples/traces/paper_example.csv \
      -e 2.0 -o verbose --visualize-ascii --stats
```

## CLI Reference

```
prove -p PROPERTY -t TRACE [options]
```

| Option | Description |
|--------|-------------|
| `-p, --property FILE` | EPLTL formula file (required) |
| `-t, --trace FILE` | Trace CSV file (required) |
| `-e, --epsilon N` | Maximum clock skew in time units (default: infinity) |
| `-o, --output MODE` | Output level: `silent`, `normal`, `verbose` (default: normal) |
| `-d, --debug N` | Debug level 0-3 (default: 0) |
| `--visualize [FILE]` | Generate graph visualization (DOT format) |
| `--visualize-ascii` | Print ASCII partial order diagram |
| `--full-graph` | Keep all graph nodes (disable pruning) |
| `--stats` | Print verification statistics |
| `--version` | Show version and exit |

**Exit codes:** `0` = property satisfied, `1` = property violated, `2` = error

### Verbose Output (`-o verbose`)

The verbose output mode prints structured, tagged lines showing the full verification pipeline:

```
[INFO] Loaded 7 events from 2 processes
[INFO] Processes: Client, Server
[INFO] Epsilon: 0.0
[INFO] Verifying formula: response -> (TRUE S request)
[EVENT] iota_Client @ process Client, props: idle
[EVENT] iota_Server @ process Server, props: idle
[EVENT] c_send @ process Client, props: request
[EVENT] s_recv @ process Server, props: busy
[EVENT] s_process @ process Server, props: processing
[EVENT] s_send @ process Server, props: response
[EVENT] c_recv @ process Client, props: satisfied
[FRONTIER] Maximal state: {Client: c_recv, Server: s_send}
SATISFIED: Property holds for at least one linearization
```

| Tag | Meaning |
|-----|---------|
| `[INFO]` | Loaded trace metadata, process list, epsilon, and formula |
| `[EVENT]` | Per-event progress showing process and active propositions |
| `[FRONTIER]` | The maximal frontier (last event per process) at verification end |

### ASCII Visualization (`--visualize-ascii`)

The `--visualize-ascii` flag prints a timeline diagram of the partial order execution, making causal relationships between events easy to understand at a glance.

```bash
prove -p formula.prop -t trace.csv -e 2.0 --visualize-ascii
```

#### How to Read the Diagram

The diagram has three sections: **header**, **timeline**, and **footer**.

**Header** -- Shows process names as column headers, a separator line, and the epsilon value:

```
      Client          Server
────────────────────────────────
(epsilon = 2.0)
```

**Timeline** -- Events are placed in their process column, ordered top-to-bottom by timestamp:

```
    iota_Client      iota_Server
      (t=0.0)          (t=0.0)
        |                |          <-- intra-process arrows
        v                v
      c_send  {request}  s_recv  {busy}
      (t=1.0)          (t=2.0)
```

Each event shows:
- **Event name** centered in its process column
- **Propositions** in curly braces next to the name (e.g., `{request}`)
- **Timestamp** below the name (e.g., `(t=1.0)`)
- **Vertical arrows** (`|` and `v`) connecting consecutive events on the same process

**Cross-process ordering arrows** appear between event rows when one event on a process is ordered before an event on another process. These arrows show **why** the ordering exists:

```
        c_send
       (t=1.0)
          ╰── c_send ≺ s_recv (VC) ──→
                              s_recv
                             (t=2.0)
```

| Arrow | Direction |
|-------|-----------|
| `╰──...──→` | Left-to-right: source process is left of target |
| `←──...──╯` | Right-to-left: source process is right of target |

The annotation between the arrows explains the ordering reason:

| Annotation | Meaning |
|------------|---------|
| `(VC)` | **Vector clock ordering** -- the source event's vector clock is strictly less than the target's (typically from message send/receive pairs) |
| `(dt=X.X>eps=Y.Y)` | **Epsilon ordering** -- the timestamp difference exceeds the clock skew bound, so the events are definitely ordered by time |
| `(VC, dt=X.X>eps=Y.Y)` | Both orderings apply simultaneously |

**Footer** -- Lists all cross-process orderings as a summary:

```
────────────────────────────────
Cross-process orderings:
  c_send ≺ s_recv  (VC)
  s_send ≺ c_recv  (VC)
```

#### Full Example

Given a two-process client-server trace with message passing (`epsilon = 0.0`):

```
         Client              Server
──────────────────────────────────────────
(epsilon = 0.0)

      iota_Client  {idle}   iota_Server  {idle}
        (t=0.0)               (t=0.0)
           |                     |
           v                     v
        c_send  {request}     s_recv  {busy}
        (t=1.0)               (t=2.0)
           ╰── c_send ≺ s_recv (VC) ──→
           |                     |
           v                     v
        c_recv  {satisfied}  s_send  {response}
        (t=5.0)               (t=4.0)
      ←── s_send ≺ c_recv (VC) ──╯

──────────────────────────────────────────
Cross-process orderings:
  c_send ≺ s_recv  (VC)
  s_send ≺ c_recv  (VC)
```

Reading this diagram: `c_send` on Client causally precedes `s_recv` on Server (via vector clock / message passing), and `s_send` on Server causally precedes `c_recv` on Client. Events within the same column (same process) are totally ordered top-to-bottom.

## Trace File Format

CSV with one event per row:

```csv
# system_processes: P1|P2
# epsilon: 2.0
eid,processes,vc,timestamp,props,event_type,msg_partner
iota_P1,P1,P1:1;P2:0,0.0,init,local,
iota_P2,P2,P1:0;P2:1,0.0,init,local,
e1,P1,P1:2;P2:0,1.0,ready,local,
e2,P2,P1:0;P2:2,1.5,done,local,
```

| Field | Description |
|-------|-------------|
| `eid` | Unique event identifier |
| `processes` | Process ID this event belongs to |
| `vc` | Vector clock (`P1:2;P2:1`) |
| `timestamp` | Global timestamp for epsilon-ordering |
| `props` | Propositions (`ready\|done`), may be empty |
| `event_type` | `local`, `send`, or `receive` |
| `msg_partner` | Target (send) or source (receive) process |

## Property File Format (EPLTL)

One formula per file. Lines starting with `#` are comments.

| Operator | Syntax | Description |
|----------|--------|-------------|
| True | `TRUE`, `true` | Always true |
| False | `FALSE`, `false` | Always false |
| Negation | `!`, `not` | Logical NOT |
| Yesterday | `@`, `Y` | Value in previous state |
| Conjunction | `&`, `&&`, `and` | Logical AND |
| Disjunction | `\|`, `\|\|`, `or` | Logical OR |
| Implication | `->`, `implies` | If-then |
| Biconditional | `<->`, `iff` | If and only if |
| Since | `S`, `since` | Since temporal operator |

**Common patterns:**

```
# Once (eventually in past): TRUE S phi
TRUE S request

# Historically (always in past): !(TRUE S !phi)
!(TRUE S !valid)

# Response: request implies past event
response -> (TRUE S request)

# Safety with Since
done -> (confirmed S ready)
```

## Python API

```python
from pathlib import Path
from prove.core.monitor import EPLTLMonitor
from prove.parser.formula import parse_formula

# From files
monitor = EPLTLMonitor.from_files(
    property_file=Path("formula.prop"),
    trace_file=Path("trace.csv"),
    epsilon=2.0,
)
result = monitor.run_from_trace()

# Programmatic
formula = parse_formula("done -> (confirmed S ready)")
monitor = EPLTLMonitor(
    formula=formula,
    processes={"P1", "P2"},
    epsilon=2.0,
)
result = monitor.run(events)

print(result.satisfied)   # True/False
print(result.verdict)     # Human-readable verdict
print(result.statistics)  # Verification statistics
```

## Project Structure

```
prove/
├── cli.py                   # Command-line interface
├── core/                    # Core monitoring engine
│   ├── vector_clock.py      # Fidge-Mattern vector clocks
│   ├── event.py             # Event representation
│   ├── partial_order.py     # Partial order computation
│   ├── frontier.py          # Frontier (global state)
│   ├── cut.py               # History-closed event sets
│   ├── sliding_window.py    # Sliding window graph
│   ├── summary.py           # EPLTL summary evaluation
│   ├── monitor.py           # Main monitor orchestration
│   ├── clock_drift.py       # Epsilon-based ordering
│   └── message_queue.py     # Message queue tracking
├── parser/                  # Formula parsing
│   ├── lexer.py             # Lexical analyzer
│   ├── grammar.py           # EPLTL grammar (SLY-based)
│   ├── ast_nodes.py         # AST node definitions
│   └── formula.py           # Formula utilities
└── utils/                   # Utilities
    ├── trace_reader.py      # CSV trace file parser
    ├── logger.py             # Structured logging
    └── visualization.py     # Graph visualization
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=prove --cov-report=html

# Run specific component tests
pytest tests/unit/test_vector_clock.py -v
```

## License

MIT
