#!/usr/bin/env bash
# =============================================================================
# Experiment 2: Mutual Exclusion
#
# Property: !(TRUE S (idle & scanning))
# Operators: Negation + Since + Conjunction
# Expected: Always VIOL (multi-drone state overlap)
#
# Prerequisites:
#   brew install gnu-time coreutils
#   source .venv/bin/activate
#
# Usage (from project root):
#   bash experiments/drone_swarm/exp2_mutual_exclusion/run_experiment.sh
# =============================================================================

set -euo pipefail

VALIDATE_TRACES=false
for arg in "$@"; do
    case "$arg" in
        --validate-traces) VALIDATE_TRACES=true ;;
    esac
done

EXP_NAME="Mutual Exclusion"
PROP_FILE_NAME="mutual_exclusion"
PROP_FORMULA='!(TRUE S (idle & scanning))'
PROP_EXPECTED="Always VIOL (multi-drone state overlap)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRACES_DIR="$SCRIPT_DIR/traces"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_FILE="$SCRIPT_DIR/results.md"
PROP_FILE="$SCRIPT_DIR/${PROP_FILE_NAME}.prop"
GENERATOR="$PROJECT_ROOT/tools/generate_drone_trace.py"
SUMMARY_LOG="$LOGS_DIR/summary.log"

TIMEOUT_SEC=1000

# Trace generation parameters
DRONES=3
MESSAGE_RATE=0.3
FRAGILE_DELTA=2.0
SEED=42

TRACE_SIZES="1000 5000 10000 100000"
TRACE_LABELS="1K 5K 10K 100K"
EPSILONS="0.0 0.5 1.0 2.0 5.0 inf"

# --- Check prerequisites ---------------------------------------------------

if ! command -v gtime &> /dev/null; then
    echo "ERROR: gtime not found. Install with: brew install gnu-time"
    exit 2
fi

if ! command -v gtimeout &> /dev/null; then
    echo "ERROR: gtimeout not found. Install with: brew install coreutils"
    exit 2
fi

if ! python -c "import prove" &> /dev/null; then
    echo "ERROR: prove package not importable. Run: pip install -e \".[dev]\""
    exit 2
fi

if [ ! -f "$PROP_FILE" ]; then
    echo "ERROR: Property file not found: $PROP_FILE"
    exit 2
fi

# --- Generate traces (skip if already exist) --------------------------------

sizes_arr=($TRACE_SIZES)
labels_arr=($TRACE_LABELS)

echo "============================================"
echo "  Experiment: ${EXP_NAME}"
echo "  Property:   ${PROP_FORMULA}"
echo "============================================"
echo ""

mkdir -p "$TRACES_DIR"
traces_generated=0

for idx in $(seq 0 $((${#sizes_arr[@]} - 1))); do
    size=${sizes_arr[$idx]}
    label=${labels_arr[$idx]}
    trace_file="$TRACES_DIR/trace_${label}.csv"

    if [ -f "$trace_file" ]; then
        echo "  Trace $label already exists, skipping generation."
    else
        echo -n "  Generating trace $label ($size events) ... "
        python "$GENERATOR" \
            --events "$size" \
            --drones "$DRONES" \
            --message-rate "$MESSAGE_RATE" \
            --fragile-delta "$FRAGILE_DELTA" \
            --seed "$SEED" \
            --output "$trace_file" 2>/dev/null
        echo "done"
        traces_generated=$((traces_generated + 1))
    fi
done

if [ "$VALIDATE_TRACES" = true ]; then
    echo ""
    echo "  Validating traces..."
    for idx in $(seq 0 $((${#labels_arr[@]} - 1))); do
        label=${labels_arr[$idx]}
        trace_file="$TRACES_DIR/trace_${label}.csv"
        echo -n "    [$label] "
        python -m prove.utils.trace_validator "$trace_file" 2>&1 | tail -1
    done
fi

echo ""

# --- Run experiment ---------------------------------------------------------

rm -rf "$LOGS_DIR"
mkdir -p "$LOGS_DIR"

echo "============================================" > "$SUMMARY_LOG"
echo "  Experiment: ${EXP_NAME}" >> "$SUMMARY_LOG"
echo "  Property:   ${PROP_FORMULA}" >> "$SUMMARY_LOG"
echo "  Started:    $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_LOG"
echo "============================================" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

eps_arr=($EPSILONS)
n_labels=${#labels_arr[@]}
n_eps=${#eps_arr[@]}
total_combos=$((n_labels * n_eps))
combo=0

RESULTS_TMP=$(mktemp)
GTIME_TMP=$(mktemp)
trap "rm -f $RESULTS_TMP $GTIME_TMP" EXIT

for label in "${labels_arr[@]}"; do
    trace_file="$TRACES_DIR/trace_${label}.csv"

    for eps in $EPSILONS; do
        combo=$((combo + 1))
        printf "  [%d/%d] %s x eps=%s ... " "$combo" "$total_combos" "$label" "$eps"

        log_file="$LOGS_DIR/${label}_eps${eps}.log"

        # Build PROVE command
        if [ "$eps" = "inf" ]; then
            prove_cmd="python -m prove -p $PROP_FILE -t $trace_file -o normal --stats"
        else
            prove_cmd="python -m prove -p $PROP_FILE -t $trace_file -e $eps -o normal --stats"
        fi

        # Run with gtime + gtimeout
        output=$(gtime -f "%e %M %P" -o "$GTIME_TMP" \
            gtimeout "$TIMEOUT_SEC" bash -c "$prove_cmd || true" 2>&1)
        run_exit=$?

        # Write per-run log
        echo "# PROVE Run: $label x eps=$eps" > "$log_file"
        echo "# Command: $prove_cmd" >> "$log_file"
        echo "# Timeout: ${TIMEOUT_SEC}s" >> "$log_file"
        echo "" >> "$log_file"
        echo "$output" >> "$log_file"

        # Read gtime metrics (tail -1 to skip "Command exited..." line)
        gtime_line=$(tail -1 "$GTIME_TMP" 2>/dev/null || echo "? ? ?")
        wall_time=$(echo "$gtime_line" | awk '{print $1}')
        max_rss_kb=$(echo "$gtime_line" | awk '{print $2}')
        cpu_pct=$(echo "$gtime_line" | awk '{print $3}')

        # Convert RSS from KB to MB
        if [ -n "$max_rss_kb" ] && [ "$max_rss_kb" != "?" ]; then
            max_rss_mb=$(python -c "print(f'{int($max_rss_kb) / 1024:.1f}')" 2>/dev/null || echo "?")
        else
            max_rss_mb="?"
        fi

        wall_time=${wall_time:-"?"}
        cpu_pct=${cpu_pct:-"?"}

        # Append gtime metrics to per-run log
        echo "" >> "$log_file"
        echo "# gtime: wall=${wall_time}s, rss=${max_rss_mb}MB, cpu=${cpu_pct}" >> "$log_file"

        # Check for timeout (gtimeout returns 124 when it kills the process)
        if [ "$run_exit" -eq 124 ]; then
            verdict="TIMEOUT"
            node_count="-"
            nodes_removed="-"
            max_summaries="-"
            events_processed="-"

            summary_line="TIMEOUT (exceeded ${TIMEOUT_SEC}s limit)"
            echo "# TIMEOUT: exceeded ${TIMEOUT_SEC}s limit" >> "$log_file"
        else
            # Parse verdict
            verdict="ERR"
            if echo "$output" | grep -q "SATISFIED"; then
                verdict="SAT"
            elif echo "$output" | grep -q "VIOLATED"; then
                verdict="VIOL"
            fi

            # Parse PROVE statistics
            node_count=$(echo "$output" | grep "Node Count:" | awk '{print $NF}')
            nodes_removed=$(echo "$output" | grep "Nodes Removed:" | awk '{print $NF}')
            max_summaries=$(echo "$output" | grep "Max Summaries:" | awk '{print $NF}')
            events_processed=$(echo "$output" | grep "Events Processed:" | awk '{print $NF}')

            node_count=${node_count:-"?"}
            nodes_removed=${nodes_removed:-"?"}
            max_summaries=${max_summaries:-"?"}
            events_processed=${events_processed:-"?"}

            summary_line=$(printf "%s (nodes=%s, time=%ss, mem=%sMB, cpu=%s)" \
                "$verdict" "$node_count" "$wall_time" "$max_rss_mb" "$cpu_pct")
        fi

        printf "%s\n" "$summary_line"
        printf "  [%d/%d] %s x eps=%s ... %s\n" \
            "$combo" "$total_combos" "$label" "$eps" "$summary_line" >> "$SUMMARY_LOG"

        echo "| $label | $eps | $verdict | $node_count | $nodes_removed | $max_summaries | $events_processed | $wall_time | $max_rss_mb | $cpu_pct |" >> "$RESULTS_TMP"
    done
done

echo ""
echo "" >> "$SUMMARY_LOG"

# --- Write results table ----------------------------------------------------

echo "--- Writing results ---"

cat > "$RESULTS_FILE" << HEADER
# Experiment 2: ${EXP_NAME}

## Property

| Formula | Operators | Expected |
|---------|-----------|----------|
| \`${PROP_FORMULA}\` | Negation + Since + Conjunction | ${PROP_EXPECTED} |

## Configuration

- **Drones**: 3 (Drone1, Drone2, Drone3)
- **Message rate**: 30%
- **Fragile delta**: 2.0
- **Seed**: 42
- **Timeout**: ${TIMEOUT_SEC}s per run
- **Epsilons**: ${EPSILONS}

## Results

| Trace | Epsilon | Verdict | Nodes | Removed | Max Sum | Events | Time (s) | Mem (MB) | CPU |
|-------|---------|---------|-------|---------|---------|--------|----------|----------|-----|
HEADER

cat "$RESULTS_TMP" >> "$RESULTS_FILE"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "" >> "$RESULTS_FILE"
echo "_Generated on ${TIMESTAMP}_" >> "$RESULTS_FILE"

echo "  Results: $RESULTS_FILE"
echo "  Summary: $SUMMARY_LOG"
echo "  Logs:    $LOGS_DIR/"
echo ""
echo "============================================"
echo "  Experiment complete!"
echo "============================================"

echo "============================================" >> "$SUMMARY_LOG"
echo "  Experiment complete: ${TIMESTAMP}" >> "$SUMMARY_LOG"
echo "============================================" >> "$SUMMARY_LOG"
