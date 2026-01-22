#!/bin/bash
#
# Delocalized Spectral Analysis - tmux Runner
# ============================================
#
# This script runs the spectral analysis experiments in a detached tmux session,
# ensuring they continue even if the SSH connection drops.
#
# Usage:
#   ./run_sweep.sh              # Run all experiments
#   ./run_sweep.sh --quick      # Quick test with 100 files
#   ./run_sweep.sh --attach     # Attach to existing session
#   ./run_sweep.sh --kill       # Kill existing session
#
# Session name: spectral_analysis
# Log file: results/experiment_run.log
#
# Author: Claude Code (Anthropic)
# Date: 2026-01-22

set -e

# Configuration
SESSION_NAME="spectral_analysis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/home/georgi/Spectral_Superposition/public/dynamics/venv"
PYTHON="${VENV_PATH}/bin/python"
LOG_FILE="${SCRIPT_DIR}/results/experiment_run.log"

# Ensure results directory exists
mkdir -p "${SCRIPT_DIR}/results"

# Parse arguments
QUICK_MODE=false
ATTACH_MODE=false
KILL_MODE=false
EXPERIMENTS=""
MAX_FILES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            MAX_FILES="100"
            shift
            ;;
        --attach)
            ATTACH_MODE=true
            shift
            ;;
        --kill)
            KILL_MODE=true
            shift
            ;;
        --experiments|-e)
            EXPERIMENTS="$2"
            shift 2
            ;;
        --max-files|-n)
            MAX_FILES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Kill existing session
if [ "$KILL_MODE" = true ]; then
    echo "Killing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || echo "No session to kill"
    exit 0
fi

# Attach to existing session
if [ "$ATTACH_MODE" = true ]; then
    echo "Attaching to tmux session: $SESSION_NAME"
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists!"
    echo "Use --attach to attach or --kill to terminate"
    exit 1
fi

# Build command
CMD="cd ${SCRIPT_DIR} && ${PYTHON} run_all_experiments.py"

if [ -n "$EXPERIMENTS" ]; then
    CMD="${CMD} --experiments ${EXPERIMENTS}"
fi

if [ -n "$MAX_FILES" ]; then
    CMD="${CMD} --max-files ${MAX_FILES}"
fi

# Add logging redirect
CMD="${CMD} 2>&1 | tee -a ${LOG_FILE}"

# Print info
echo "========================================"
echo "Delocalized Spectral Analysis"
echo "========================================"
echo "Session name: $SESSION_NAME"
echo "Python: $PYTHON"
echo "Working dir: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Command:"
echo "  $CMD"
echo ""

# Create tmux session
echo "Creating detached tmux session..."
tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo "========================================"
echo "Session started successfully!"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To kill session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo "========================================"
