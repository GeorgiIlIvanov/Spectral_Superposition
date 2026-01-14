#!/bin/bash
# Spectral Superposition Sweep v2 - Launcher Script
# Optimized for 8x L4 GPUs on GCP g2-standard-96

set -e

# === Configuration ===
RESULTS_DIR="${1:-results_v2}"
NUM_GPUS="${2:-8}"
WORKERS_PER_GPU="${3:-4}"

# === Colors for output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Spectral Superposition Sweep v2${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# === Check prerequisites ===
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}Error: PyTorch not found. Install with: pip install torch${NC}"
    exit 1
fi

# Check h5py
if ! python -c "import h5py" 2>/dev/null; then
    echo -e "${RED}Error: h5py not found. Install with: pip install h5py${NC}"
    exit 1
fi

# Check CUDA
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo -e "${RED}Error: CUDA not available${NC}"
    exit 1
fi

# Get actual GPU count
ACTUAL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo -e "Found ${GREEN}$ACTUAL_GPUS${NC} GPUs"

if [ "$ACTUAL_GPUS" -lt "$NUM_GPUS" ]; then
    echo -e "${YELLOW}Warning: Requested $NUM_GPUS GPUs but only $ACTUAL_GPUS available${NC}"
    NUM_GPUS=$ACTUAL_GPUS
fi

# === Setup directories ===
mkdir -p "$RESULTS_DIR"
mkdir -p logs

# === Display configuration ===
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Results directory: $RESULTS_DIR"
echo "  GPUs: $NUM_GPUS"
echo "  Workers per GPU: $WORKERS_PER_GPU"
echo "  Total workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo ""

# === Experiment stats ===
TOTAL_EXPERIMENTS=$((32 * 50 * 2))  # M_VALUES * S_VALUES * SEEDS
EXPERIMENTS_PER_WORKER=$((TOTAL_EXPERIMENTS / (NUM_GPUS * WORKERS_PER_GPU)))

echo -e "${YELLOW}Experiment Grid:${NC}"
echo "  M values: 32 (16 to 512)"
echo "  S values: 50 (0.0 to 0.99)"
echo "  Seeds: 2"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  ~Experiments per worker: $EXPERIMENTS_PER_WORKER"
echo ""

# === Check for existing results ===
EXISTING_COUNT=$(find "$RESULTS_DIR" -name "*.h5" 2>/dev/null | wc -l)
if [ "$EXISTING_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Found $EXISTING_COUNT existing results - will resume${NC}"
fi

# === Launch ===
echo -e "${GREEN}Starting sweep...${NC}"
echo "Logs: logs/sweep.log"
echo ""

# Run with nohup for background execution
LOG_FILE="logs/sweep_$(date +%Y%m%d_%H%M%S).log"

python sweep.py \
    --gpus "$NUM_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_FILE"

# === Completion ===
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Sweep completed!${NC}"
echo -e "${GREEN}============================================${NC}"

# Final count
FINAL_COUNT=$(find "$RESULTS_DIR" -name "*.h5" 2>/dev/null | wc -l)
echo -e "Results: ${GREEN}$FINAL_COUNT${NC} / $TOTAL_EXPERIMENTS"

# Storage usage
if [ "$FINAL_COUNT" -gt 0 ]; then
    STORAGE=$(du -sh "$RESULTS_DIR" | cut -f1)
    echo -e "Storage: ${GREEN}$STORAGE${NC}"
fi

echo ""
echo "To verify: python verify_results.py --results-dir $RESULTS_DIR"
