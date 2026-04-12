#!/bin/bash
# ============================================================================
# Ablation Study Runner for AGEA Hyperparameter Analysis
# ============================================================================
# Uses run_agea.py and launches non-default ablations only.
# Defaults in run_agea.py are intentionally skipped:
#   initial-epsilon=0.3, epsilon-decay=0.98, min-epsilon=0.05,
#   novelty-threshold=0.15, novelty-window=5,
#   novelty-threshold-mode=adaptive, query-method=local,
#   graph-filter=enabled, success-rate-detection=enabled
#
# Groups:
#   A: Epsilon decay (2 runs)
#   B: Novelty threshold (3 runs)
#   C: Minimum epsilon (3 runs)
#   D: Novelty window (3 runs)
#   E: Novelty-threshold mode (1 run: fixed)
#   F: Graph filter module (1 run: disabled)
#   G: Success-rate detection (1 run: disabled)
#   H: Initial epsilon (2 runs)
#   I: Query method (2 runs: global/basic)
#
# Usage:
#   bash run_ablation_study.sh              # Run all groups in parallel
#   bash run_ablation_study.sh A            # Run only Group A
#   bash run_ablation_study.sh B            # Run only Group B
#   bash run_ablation_study.sh C            # Run only Group C
#   bash run_ablation_study.sh D E F        # Run selected groups
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/medical/ablation_logs"
mkdir -p "$LOG_DIR"

# Common arguments for all experiments
COMMON_ARGS="--dataset medical --turns 250 --query-method local"

# Default hyperparameters (baseline values)
DEFAULT_EPSILON=0.3
DEFAULT_DECAY=0.98
DEFAULT_MIN_EPSILON=0.05
DEFAULT_THRESHOLD=0.15
DEFAULT_NOVELTY_WINDOW=5

run_experiment() {
    local name="$1"
    local output_suffix="$2"
    shift 2
    local log_file="$LOG_DIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $name"
    python run_agea.py \
        $COMMON_ARGS \
        --initial-epsilon "$DEFAULT_EPSILON" \
        --output-suffix "$output_suffix" \
        "$@" \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE: $name (success)"
    else
        echo "[$(date '+%H:%M:%S')] FAIL: $name (exit code $exit_code, see $log_file)"
    fi
    return $exit_code
}

# ---- Group A: Epsilon Decay Rate ----
run_group_a() {
    echo "========== Group A: Epsilon Decay Rate =========="
    run_experiment "decay_0.90" \
        "ablation_study_decay_0.90" \
        --epsilon-decay 0.90 \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" &

    run_experiment "decay_0.99" \
        "ablation_study_decay_0.99" \
        --epsilon-decay 0.99 \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" &
}

# ---- Group B: Novelty Threshold ----
run_group_b() {
    echo "========== Group B: Novelty Threshold =========="
    run_experiment "threshold_0.10" \
        "ablation_study_threshold_0.10" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold 0.10 &

    run_experiment "threshold_0.25" \
        "ablation_study_threshold_0.25" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold 0.25 &

    run_experiment "threshold_0.40" \
        "ablation_study_threshold_0.40" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold 0.40 &
}

# ---- Group C: Minimum Epsilon ----
run_group_c() {
    echo "========== Group C: Minimum Epsilon =========="
    run_experiment "min_eps_0.01" \
        "ablation_study_min_eps_0.01" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon 0.01 \
        --novelty-threshold "$DEFAULT_THRESHOLD" &

    run_experiment "min_eps_0.10" \
        "ablation_study_min_eps_0.10" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon 0.10 \
        --novelty-threshold "$DEFAULT_THRESHOLD" &

    run_experiment "min_eps_0.20" \
        "ablation_study_min_eps_0.20" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon 0.20 \
        --novelty-threshold "$DEFAULT_THRESHOLD" &
}

# ---- Group D: Novelty Window ----
run_group_d() {
    echo "========== Group D: Novelty Window =========="
    run_experiment "novelty_window_1" \
        "ablation_study_novelty_window_1" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window 1 &

    run_experiment "novelty_window_10" \
        "ablation_study_novelty_window_10" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window 10 &

    run_experiment "novelty_window_20" \
        "ablation_study_novelty_window_20" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window 20 &
}

# ---- Group E: Novelty Threshold Mode ----
run_group_e() {
    echo "========== Group E: Novelty Threshold Mode =========="
    # Default mode is adaptive, so only run fixed as ablation.
    run_experiment "threshold_mode_fixed" \
        "ablation_study_threshold_mode_fixed" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" \
        --novelty-threshold-mode fixed &
}

# ---- Group F: Graph Filter Module ----
run_group_f() {
    echo "========== Group F: Graph Filter Module =========="
    # Default is enabled; ablation disables it.
    run_experiment "graph_filter_disabled" \
        "ablation_study_graph_filter_disabled" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" \
        --disable-graph-filter &
}

# ---- Group G: Success-Rate Detection ----
run_group_g() {
    echo "========== Group G: Success-Rate Detection =========="
    # Default is enabled; ablation disables it.
    run_experiment "success_rate_detection_off" \
        "ablation_study_success_rate_detection_off" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" \
        --disable-success-rate-detection &
}

# ---- Group H: Initial Epsilon ----
run_group_h() {
    echo "========== Group H: Initial Epsilon =========="
    run_experiment "initial_eps_0.10" \
        "ablation_study_initial_eps_0.10" \
        --initial-epsilon 0.10 \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" &

    run_experiment "initial_eps_0.20" \
        "ablation_study_initial_eps_0.20" \
        --initial-epsilon 0.20 \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" &
}

# ---- Group I: Query Method ----
run_group_i() {
    echo "========== Group I: Query Method =========="
    # Default is local; ablations run global/basic.
    run_experiment "query_method_global" \
        "ablation_study_query_method_global" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" \
        --query-method global &

    run_experiment "query_method_basic" \
        "ablation_study_query_method_basic" \
        --epsilon-decay "$DEFAULT_DECAY" \
        --min-epsilon "$DEFAULT_MIN_EPSILON" \
        --novelty-threshold "$DEFAULT_THRESHOLD" \
        --novelty-window "$DEFAULT_NOVELTY_WINDOW" \
        --query-method basic &
}

# ---- Main ----
echo "============================================================"
echo "  AGEA Ablation Study (non-default configurations)"
echo "  Target runner: run_agea.py"
echo "  Logs: $LOG_DIR/"
echo "============================================================"

GROUPS="${@:-A B C D E F G H I}"

for group in $GROUPS; do
    case "${group^^}" in
        A) run_group_a ;;
        B) run_group_b ;;
        C) run_group_c ;;
        D) run_group_d ;;
        E) run_group_e ;;
        F) run_group_f ;;
        G) run_group_g ;;
        H) run_group_h ;;
        I) run_group_i ;;
        *) echo "Unknown group: $group (use A-I)" ;;
    esac
done

echo ""
echo "All experiments launched. Waiting for completion..."
wait
echo ""
echo "============================================================"
echo "  All experiments finished. Check logs at: $LOG_DIR/"
echo "  Results at: $SCRIPT_DIR/medical/*/*/ablation_study_*/"
echo "============================================================"
