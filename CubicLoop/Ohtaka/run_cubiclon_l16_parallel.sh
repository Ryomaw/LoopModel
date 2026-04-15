#!/bin/bash
#
# SLURM Job Script for Parallel Monte Carlo Simulation
# Cubic Loop O(n) Model - Large Scale Parameter Sweep
#
# Execution on Ohtaka Supercomputer
# Optimized for L=16, n=1..100, x=0.1..5.0 with multiple ts/os configs
#
# Estimated total time: ~24-48 hours depending on load
#

#SBATCH --job-name=CubicLOn_L16_parallel
#SBATCH --partition=F1cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=230G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=watanabe-ryoma@issp.u-tokyo.ac.jp

set -e

# ============================================================================
# Configuration
# ============================================================================

# Directory setup
WORK_DIR="/work/users/$(whoami)/CubicLOn_L16"
RESULT_DIR="${WORK_DIR}/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python environment
MODULE_LOAD="module load python/3.11"
VENV_PATH="${WORK_DIR}/venv"

# Job parameters
LATTICE_SIZE=16
NUM_WORKERS=32  # Match cpus-per-task
OUTPUT_PREFIX="CubicLOn_L${LATTICE_SIZE}"

# ============================================================================
# Setup
# ============================================================================

echo "=========================================================================="
echo "Monte Carlo Simulation: Cubic Loop O(n) Model"
echo "=========================================================================="
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Hostname: $(hostname)"
echo "=========================================================================="

# Create work directory
mkdir -p "${WORK_DIR}"
mkdir -p "${RESULT_DIR}"

cd "${WORK_DIR}"

# Setup Python environment
if [ ! -d "${VENV_PATH}" ]; then
    echo "[INFO] Creating virtual environment..."
    ${MODULE_LOAD}
    python -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install numpy pillow
else
    source "${VENV_PATH}/bin/activate"
fi

# Copy Python script
echo "[INFO] Copying simulation script..."
cp "${SCRIPT_DIR}/CubicLOn_parallel_MC.py" "${WORK_DIR}/"

# ============================================================================
# Run Simulation
# ============================================================================

echo ""
echo "=========================================================================="
echo "Simulation Configuration"
echo "=========================================================================="
echo "Lattice size L: ${LATTICE_SIZE}"
echo "n values: 1 2 3 4 5 6 7 8 9 10 20 50 100"
echo "x range: 0.1 to 5.0 (step 0.1) = 49 points"
echo "ts/os configs: (10000,40000) (40000,40000) (10000,160000) (40000,160000)"
echo "Total simulations: 13 × 49 × 4 = 2548"
echo "Parallel workers: ${NUM_WORKERS}"
echo "Output directory: ${RESULT_DIR}"
echo "=========================================================================="
echo ""

python CubicLOn_parallel_MC.py \
    --lattice_size ${LATTICE_SIZE} \
    --n_values 1 2 3 4 5 6 7 8 9 10 20 50 100 \
    --x_min 0.1 \
    --x_max 5.0 \
    --x_step 0.1 \
    --output_dir "${RESULT_DIR}" \
    --num_workers ${NUM_WORKERS}

SCRIPT_EXIT=$?

# ============================================================================
# Post-processing and Summary
# ============================================================================

echo ""
echo "=========================================================================="
echo "Simulation Complete"
echo "=========================================================================="
echo "End time: $(date)"

if [ ${SCRIPT_EXIT} -eq 0 ]; then
    echo "Status: SUCCESS"
    
    # List output files
    echo ""
    echo "Output files:"
    ls -lh "${RESULT_DIR}/" | tail -n +2
    
    # Summary statistics
    if [ -f "${RESULT_DIR}/results.csv" ]; then
        LINES=$(wc -l < "${RESULT_DIR}/results.csv")
        echo ""
        echo "CSV Results: ${LINES} lines (including header)"
        head -5 "${RESULT_DIR}/results.csv"
        echo "..."
    fi
else
    echo "Status: FAILED (exit code ${SCRIPT_EXIT})"
    exit ${SCRIPT_EXIT}
fi

echo "=========================================================================="
