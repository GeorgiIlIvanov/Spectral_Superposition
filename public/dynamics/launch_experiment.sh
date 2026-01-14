#!/bin/bash
# ============================================================
# BULLETPROOF EXPERIMENT LAUNCHER
# Survives SSH disconnection, system monitoring, auto-resume
# ============================================================

set -e

# === CONFIGURATION ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
RESULTS_DIR="${1:-$SCRIPT_DIR/results_v2}"
NUM_GPUS="${2:-8}"
WORKERS_PER_GPU="${3:-4}"

# Screen session name
SESSION_NAME="superposition_sweep"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/sweep.pid"
LOG_FILE="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"

# === COLORS ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# === FUNCTIONS ===
print_header() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  SPECTRAL SUPERPOSITION EXPERIMENT v2${NC}"
    echo -e "${CYAN}  Bulletproof Mode - Survives SSH Disconnection${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

check_disk_space() {
    local target_dir="$1"
    local required_gb="${2:-100}"  # Default 100GB requirement

    # Get parent directory that exists
    local check_dir="$target_dir"
    while [[ ! -d "$check_dir" ]]; do
        check_dir="$(dirname "$check_dir")"
    done

    local available_gb=$(df -BG "$check_dir" | awk 'NR==2 {print $4}' | tr -d 'G')

    echo -e "${YELLOW}Disk Space Check:${NC}"
    echo "  Target directory: $target_dir"
    echo "  Mount point: $(df "$check_dir" | awk 'NR==2 {print $6}')"
    echo "  Available: ${available_gb}GB"
    echo "  Required: ~${required_gb}GB"

    if [[ "$available_gb" -lt "$required_gb" ]]; then
        echo -e "${RED}ERROR: Insufficient disk space!${NC}"
        echo "  Need at least ${required_gb}GB, only ${available_gb}GB available"
        return 1
    fi

    echo -e "  Status: ${GREEN}OK${NC}"
    return 0
}

check_gpus() {
    echo -e "${YELLOW}GPU Check:${NC}"

    local PYTHON="$VENV_DIR/bin/python"

    if ! $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo -e "${RED}ERROR: CUDA not available${NC}"
        return 1
    fi

    local actual_gpus=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
    echo "  Available GPUs: $actual_gpus"

    if [[ "$actual_gpus" -lt "$NUM_GPUS" ]]; then
        echo -e "${YELLOW}  Warning: Requested $NUM_GPUS GPUs, using $actual_gpus${NC}"
        NUM_GPUS=$actual_gpus
    fi

    # Show GPU info
    $PYTHON -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1e9
    print(f'    GPU {i}: {props.name} ({mem_gb:.1f}GB)')
"
    echo -e "  Status: ${GREEN}OK${NC}"
    return 0
}

check_dependencies() {
    echo -e "${YELLOW}Dependency Check:${NC}"

    # Check venv exists
    if [[ ! -d "$VENV_DIR" ]]; then
        echo -e "${RED}ERROR: Virtual environment not found at $VENV_DIR${NC}"
        echo "  Create with: python3 -m venv venv && source venv/bin/activate && pip install torch numpy h5py tqdm"
        return 1
    fi

    local PYTHON="$VENV_DIR/bin/python"
    local missing=()

    $PYTHON -c "import torch" 2>/dev/null || missing+=("torch")
    $PYTHON -c "import h5py" 2>/dev/null || missing+=("h5py")
    $PYTHON -c "import numpy" 2>/dev/null || missing+=("numpy")

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${RED}ERROR: Missing Python packages: ${missing[*]}${NC}"
        echo "  Install with: source venv/bin/activate && pip install ${missing[*]}"
        return 1
    fi

    echo -e "  Status: ${GREEN}OK${NC}"
    return 0
}

is_running() {
    if screen -list | grep -q "$SESSION_NAME"; then
        return 0
    fi
    return 1
}

show_status() {
    echo -e "${YELLOW}Experiment Status:${NC}"

    if is_running; then
        echo -e "  Screen session: ${GREEN}RUNNING${NC} ($SESSION_NAME)"
    else
        echo -e "  Screen session: ${RED}NOT RUNNING${NC}"
    fi

    # Count results
    if [[ -d "$RESULTS_DIR" ]]; then
        local count=$(find "$RESULTS_DIR" -name "*.h5" 2>/dev/null | wc -l)
        local total=3200
        local pct=$((count * 100 / total))
        echo "  Results: $count / $total ($pct%)"

        if [[ "$count" -gt 0 ]]; then
            local storage=$(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)
            echo "  Storage used: $storage"

            # Latest file
            local latest=$(ls -t "$RESULTS_DIR"/*.h5 2>/dev/null | head -1)
            if [[ -n "$latest" ]]; then
                local age=$(( ($(date +%s) - $(stat -c %Y "$latest")) / 60 ))
                echo "  Last checkpoint: ${age}m ago"
            fi
        fi
    else
        echo "  Results: 0 / 3200 (0%)"
    fi
}

start_experiment() {
    echo -e "${GREEN}Starting experiment in detached screen session...${NC}"

    # Create directories
    mkdir -p "$RESULTS_DIR" "$LOG_DIR"

    # Create the actual runner script
    cat > "$LOG_DIR/runner.sh" << 'RUNNER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$1"
NUM_GPUS="$2"
WORKERS_PER_GPU="$3"
LOG_FILE="$4"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

echo "========================================"
echo "Experiment started at $(date)"
echo "PID: $$"
echo "Python: $(which python)"
echo "Results: $RESULTS_DIR"
echo "GPUs: $NUM_GPUS"
echo "Workers/GPU: $WORKERS_PER_GPU"
echo "========================================"

# Run with automatic restart on failure
MAX_RETRIES=5
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    python sweep.py \
        --gpus "$NUM_GPUS" \
        --workers-per-gpu "$WORKERS_PER_GPU" \
        --results-dir "$RESULTS_DIR"

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "========================================"
        echo "Experiment completed successfully at $(date)"
        echo "========================================"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "========================================"
        echo "Experiment failed with code $EXIT_CODE at $(date)"
        echo "Retry $RETRY_COUNT / $MAX_RETRIES in 30 seconds..."
        echo "========================================"
        sleep 30
    fi
done

if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
    echo "Max retries exceeded. Check logs."
fi
RUNNER_EOF

    chmod +x "$LOG_DIR/runner.sh"

    # Launch in screen
    screen -dmS "$SESSION_NAME" bash -c \
        "\"$LOG_DIR/runner.sh\" \"$RESULTS_DIR\" \"$NUM_GPUS\" \"$WORKERS_PER_GPU\" \"$LOG_FILE\" 2>&1 | tee \"$LOG_FILE\""

    sleep 2

    if is_running; then
        echo -e "${GREEN}SUCCESS: Experiment launched in screen session '$SESSION_NAME'${NC}"
        echo ""
        echo "The experiment will continue even if you disconnect SSH."
        echo ""
        echo -e "${CYAN}Useful commands:${NC}"
        echo "  Attach to session:  screen -r $SESSION_NAME"
        echo "  Detach from session: Ctrl+A, then D"
        echo "  Check status:       $0 status"
        echo "  View logs:          tail -f $LOG_FILE"
        echo "  Stop experiment:    $0 stop"
    else
        echo -e "${RED}ERROR: Failed to start screen session${NC}"
        return 1
    fi
}

stop_experiment() {
    if is_running; then
        echo -e "${YELLOW}Stopping experiment...${NC}"
        screen -S "$SESSION_NAME" -X quit
        sleep 2

        if is_running; then
            echo -e "${RED}Failed to stop gracefully, sending SIGTERM...${NC}"
            screen -S "$SESSION_NAME" -X stuff $'\003'  # Ctrl+C
            sleep 5
            screen -S "$SESSION_NAME" -X quit 2>/dev/null
        fi

        echo -e "${GREEN}Experiment stopped${NC}"
    else
        echo "No running experiment to stop"
    fi
}

# === MAIN ===
print_header

case "${1:-start}" in
    start)
        # Pre-flight checks
        echo -e "${YELLOW}Pre-flight Checks:${NC}"
        echo ""

        check_dependencies || exit 1
        echo ""

        check_gpus || exit 1
        echo ""

        check_disk_space "$RESULTS_DIR" 100 || exit 1
        echo ""

        # Check if already running
        if is_running; then
            echo -e "${YELLOW}Experiment is already running!${NC}"
            show_status
            echo ""
            echo "Use '$0 attach' to view progress"
            echo "Use '$0 stop' to stop the experiment"
            exit 0
        fi

        echo ""
        start_experiment
        ;;

    stop)
        stop_experiment
        ;;

    status)
        show_status
        ;;

    attach)
        if is_running; then
            echo "Attaching to session. Press Ctrl+A, D to detach."
            sleep 1
            screen -r "$SESSION_NAME"
        else
            echo "No running experiment to attach to"
            echo "Start with: $0 start"
        fi
        ;;

    logs)
        # Find latest log
        latest_log=$(ls -t "$LOG_DIR"/sweep_*.log 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            echo "Tailing: $latest_log"
            echo "Press Ctrl+C to exit"
            tail -f "$latest_log"
        else
            echo "No logs found"
        fi
        ;;

    verify)
        "$VENV_DIR/bin/python" "$SCRIPT_DIR/verify_results.py" --results-dir "$RESULTS_DIR"
        ;;

    *)
        echo "Usage: $0 {start|stop|status|attach|logs|verify} [results_dir] [num_gpus] [workers_per_gpu]"
        echo ""
        echo "Commands:"
        echo "  start   - Start the experiment in background (default)"
        echo "  stop    - Stop the running experiment"
        echo "  status  - Show experiment status"
        echo "  attach  - Attach to running screen session"
        echo "  logs    - Tail the latest log file"
        echo "  verify  - Verify results integrity"
        exit 1
        ;;
esac
