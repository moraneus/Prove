#!/usr/bin/env bash
# =============================================================================
# Safety_Drone experiment runner
#
# Property: relative_confirmed -> (TRUE S velocity_sent)
#
# 15 explicit prove invocations (3 trace files × 5 epsilons), strictly
# sequential. Each capped at 1000s via gtimeout; on timeout, the run is
# logged as TIMEOUT and the next invocation proceeds. Time and peak
# memory captured via gtime.
#
# This script does NOT generate trace files. The three traces must
# already exist:
#   traces/trace_1k.csv
#   traces/trace_10k.csv
#   traces/trace_100k.csv
#
# Generate them, e.g., with:
#   python tools/generate_safety_drone_trace.py -n 1000   -k 10 -s 42 \
#       -o experiments/Safety_Drone/traces/trace_1k.csv
#   python tools/generate_safety_drone_trace.py -n 10000  -k 10 -s 42 \
#       -o experiments/Safety_Drone/traces/trace_10k.csv
#   python tools/generate_safety_drone_trace.py -n 100000 -k 10 -s 42 \
#       -o experiments/Safety_Drone/traces/trace_100k.csv
#
# Prerequisites:
#   brew install gnu-time coreutils
#   source .venv/bin/activate && pip install -e ".[dev]"
#
# Usage (from project root):
#   bash experiments/Safety_Drone/run_experiment.sh
# =============================================================================
set -uo pipefail

EXP_NAME="Safety_Drone"
EXP_TAG="drone"
PROP_FILE_NAME="safety_drone"
PROP_FORMULA='relative_confirmed -> (TRUE S velocity_sent)'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACES_DIR="$SCRIPT_DIR/traces"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_FILE="$SCRIPT_DIR/results.md"
PROP_FILE="$SCRIPT_DIR/${PROP_FILE_NAME}.prop"

TIMEOUT_SEC=1000

# --- Prereq checks ---------------------------------------------------------

for tool in gtime gtimeout python; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "ERROR: '$tool' not found. Install GNU time/coreutils and activate the venv." >&2
        exit 2
    }
done
python -c "import prove" 2>/dev/null || {
    echo "ERROR: prove package not importable. Run: pip install -e \".[dev]\"" >&2
    exit 2
}
[ -f "$PROP_FILE" ] || { echo "ERROR: missing property file: $PROP_FILE" >&2; exit 2; }
for label in 1k 10k 100k; do
    [ -f "$TRACES_DIR/trace_${label}.csv" ] || {
        echo "ERROR: missing trace file: $TRACES_DIR/trace_${label}.csv" >&2
        echo "  Generate it before running this script." >&2
        exit 2
    }
done

# --- Per-run helper --------------------------------------------------------
#
# run_one <size_label> <epsilon>
#   Wraps a single explicit invocation
#       gtime gtimeout 1000 python -m prove -p <prop> -t <trace> -e <eps> --stats
#   parses output, writes a per-run log, and appends one row to the table.
#
# IMPORTANT: -e is always passed, including for ε=inf. Omitting -e would
# cause the CLI to fall back to the trace's "# epsilon: ..." directive,
# silently overriding "ε=inf" with whatever the trace declares.

rm -rf "$LOGS_DIR"; mkdir -p "$LOGS_DIR"
GTIME_TMP=$(mktemp); RESULTS_TMP=$(mktemp)
trap 'rm -f "$GTIME_TMP" "$RESULTS_TMP"' EXIT

# Ctrl+C handler. The prove invocation runs in the background (so we have
# its PID and can kill it explicitly), and gtimeout is given --foreground
# so SIGINT propagates to python through the timeout wrapper instead of
# being absorbed by gtimeout's default isolated-process-group behavior.
# Pressing Ctrl+C marks the run aborted and kills the running prove;
# two Ctrl+Cs within 2s abort the whole sweep.
SKIP_CURRENT=0
LAST_INT_TIME=0
CURRENT_BG_PID=0
handle_int() {
    local now
    now=$(date +%s)
    if [ "$LAST_INT_TIME" -ne 0 ] && [ $((now - LAST_INT_TIME)) -le 2 ]; then
        printf "\n>> Double Ctrl+C: aborting entire sweep.\n" >&2
        if [ "$CURRENT_BG_PID" -ne 0 ]; then
            pkill -KILL -P "$CURRENT_BG_PID" 2>/dev/null || true
            kill -KILL "$CURRENT_BG_PID" 2>/dev/null || true
        fi
        exit 130
    fi
    LAST_INT_TIME=$now
    SKIP_CURRENT=1
    if [ "$CURRENT_BG_PID" -ne 0 ]; then
        pkill -TERM -P "$CURRENT_BG_PID" 2>/dev/null || true
        kill  -TERM    "$CURRENT_BG_PID" 2>/dev/null || true
    fi
    printf "\n>> Ctrl+C: skipping current run (press again within 2s to abort sweep)...\n" >&2
}
trap handle_int INT

run_one() {
    SKIP_CURRENT=0
    local label=$1
    local eps=$2
    local trace_file="$TRACES_DIR/trace_${label}.csv"
    local run_id="${EXP_TAG}, ${label}, e=${eps}"
    local log_file="$LOGS_DIR/${EXP_TAG}_${label}_e${eps}.log"

    printf "  %-40s ... " "$run_id"

    # Run in the background so the SIGINT trap can kill it explicitly.
    # --foreground makes gtimeout forward SIGINT/SIGQUIT to python.
    local tmp_out
    tmp_out=$(mktemp)
    gtime -f "%e %M %P" -o "$GTIME_TMP" \
        gtimeout --foreground "$TIMEOUT_SEC" python -m prove \
            -p "$PROP_FILE" -t "$trace_file" -e "$eps" -o normal --stats \
        > "$tmp_out" 2>&1 &
    CURRENT_BG_PID=$!

    local output run_exit
    wait "$CURRENT_BG_PID" 2>/dev/null
    run_exit=$?
    CURRENT_BG_PID=0
    output=$(cat "$tmp_out")
    rm -f "$tmp_out"

    local gtime_line wall_time max_rss_kb cpu_pct max_rss_mb
    gtime_line=$(tail -1 "$GTIME_TMP" 2>/dev/null || echo "? ? ?")
    wall_time=$(echo "$gtime_line" | awk '{print $1}')
    max_rss_kb=$(echo "$gtime_line" | awk '{print $2}')
    cpu_pct=$(echo "$gtime_line" | awk '{print $3}')
    if [ -n "$max_rss_kb" ] && [ "$max_rss_kb" != "?" ]; then
        max_rss_mb=$(python -c "print(f'{int($max_rss_kb)/1024:.1f}')" 2>/dev/null || echo "?")
    else
        max_rss_mb="?"
    fi
    wall_time=${wall_time:-"?"}
    cpu_pct=${cpu_pct:-"?"}

    {
        echo "# Run: $run_id"
        echo "# Command: gtime python -m prove -p $PROP_FILE -t $trace_file -e $eps -o normal --stats"
        echo "# Timeout: ${TIMEOUT_SEC}s"
        echo ""
        echo "$output"
        echo ""
        echo "# gtime: wall=${wall_time}s, rss=${max_rss_mb}MB, cpu=${cpu_pct}"
    } > "$log_file"

    local verdict nodes removed max_sum events summary
    if [ "$SKIP_CURRENT" -eq 1 ]; then
        verdict="ABORTED"; nodes="-"; removed="-"; max_sum="-"; events="-"
        wall_time="(user)"
        echo "# ABORTED by user (Ctrl+C)" >> "$log_file"
        summary="ABORTED (user)"
    elif [ "$run_exit" -eq 124 ]; then
        verdict="TIMEOUT"; nodes="-"; removed="-"; max_sum="-"; events="-"
        wall_time=">${TIMEOUT_SEC}"
        echo "# TIMEOUT after ${TIMEOUT_SEC}s" >> "$log_file"
        summary="TIMEOUT"
    else
        verdict="ERR"
        echo "$output" | grep -q "SATISFIED" && verdict="SAT"
        echo "$output" | grep -q "VIOLATED"  && verdict="VIOL"
        nodes=$(echo   "$output" | awk '/Node Count:/      {print $NF}')
        removed=$(echo "$output" | awk '/Nodes Removed:/   {print $NF}')
        max_sum=$(echo "$output" | awk '/Max Summaries:/   {print $NF}')
        events=$(echo  "$output" | awk '/Events Processed:/{print $NF}')
        nodes=${nodes:-"?"}; removed=${removed:-"?"}
        max_sum=${max_sum:-"?"}; events=${events:-"?"}
        summary=$(printf "%s nodes=%s time=%ss mem=%sMB" \
            "$verdict" "$nodes" "$wall_time" "$max_rss_mb")
    fi

    printf "%s\n" "$summary"
    echo "| $EXP_TAG | $label | $eps | $verdict | $nodes | $removed | $max_sum | $events | $wall_time | $max_rss_mb | $cpu_pct |" \
        >> "$RESULTS_TMP"
}

# --- Banner ----------------------------------------------------------------

echo "============================================"
echo "  Experiment: ${EXP_NAME}"
echo "  Property:   ${PROP_FORMULA}"
echo "  Timeout:    ${TIMEOUT_SEC}s per run"
echo "============================================"

# --- 15 explicit runs ------------------------------------------------------

run_one 1k   0.0
run_one 1k   0.5
run_one 1k   1.0
run_one 1k   5.0
run_one 1k   inf

run_one 10k  0.0
run_one 10k  0.5
run_one 10k  1.0
run_one 10k  5.0
run_one 10k  inf

run_one 100k 0.0
run_one 100k 0.5
run_one 100k 1.0
run_one 100k 5.0
run_one 100k inf

echo ""

# --- Write results.md ------------------------------------------------------

cat > "$RESULTS_FILE" << HEADER
# ${EXP_NAME} — Sweep Results

**Property**: \`${PROP_FORMULA}\`

**Sweep**: 15 explicit runs over trace sizes {1k, 10k, 100k} × ε ∈ {0.0, 0.5, 1.0, 5.0, inf}
**Timeout**: ${TIMEOUT_SEC}s per run (TIMEOUT entries indicate the run was killed)

| Exp | Trace | ε | Verdict | Nodes | Removed | MaxSum | Events | Time (s) | Mem (MB) | CPU |
|-----|-------|---|---------|-------|---------|--------|--------|----------|----------|-----|
HEADER
cat "$RESULTS_TMP" >> "$RESULTS_FILE"
{
    echo ""
    echo "_Generated $(date '+%Y-%m-%d %H:%M:%S')_"
} >> "$RESULTS_FILE"

echo "Results: $RESULTS_FILE"
echo "Logs:    $LOGS_DIR/"
