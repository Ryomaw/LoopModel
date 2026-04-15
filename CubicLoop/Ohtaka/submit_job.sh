#!/bin/bash
#
# Quick Job Submission Helper
# Easily submit the Monte Carlo simulation to Ohtaka
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_cubiclon_l16_parallel.sh"

echo "=========================================================================="
echo "Monte Carlo Simulation Submission Helper"
echo "=========================================================================="
echo ""

if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "[ERROR] Job script not found: ${JOB_SCRIPT}"
    exit 1
fi

# Check SBATCH availability
if ! command -v sbatch &> /dev/null; then
    echo "[ERROR] sbatch command not found. Make sure you're on an Ohtaka login node."
    exit 1
fi

echo "Job script: ${JOB_SCRIPT}"
echo ""

# Menu
echo "Select submission option:"
echo ""
echo "  1) Standard run (all 2548 tasks, ~7 hours)"
echo "  2) Quick run (subset for testing, ~2 hours)"
echo "  3) Custom parameters (interactive)"
echo "  4) View job status"
echo "  5) Exit"
echo ""

read -p "Enter choice [1-5]: " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "[INFO] Submitting standard job..."
        sbatch "${JOB_SCRIPT}"
        echo "[INFO] Job submitted successfully!"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u \$(whoami)"
        echo "or"
        echo "  tail -f CubicLOn_L16_parallel_*.log"
        ;;
    2)
        echo ""
        echo "[INFO] Creating and submitting quick test job..."
        QUICK_SCRIPT="${SCRIPT_DIR}/run_cubiclon_l16_quick.sh"
        
        # Create quick script (copy with modifications)
        cp "${JOB_SCRIPT}" "${QUICK_SCRIPT}"
        
        # Modify parameters for quick run
        sed -i 's/#SBATCH --job-name=CubicLOn_L16_parallel/#SBATCH --job-name=CubicLOn_L16_quick/' "${QUICK_SCRIPT}"
        sed -i 's/#SBATCH --time=72:00:00/#SBATCH --time=04:00:00/' "${QUICK_SCRIPT}"
        sed -i 's/--cpus-per-task=32/--cpus-per-task=16/' "${QUICK_SCRIPT}"
        
        # Add custom parameters to quick script
        sed -i 's/--n_values 1 2 3 4 5 6 7 8 9 10 20 50 100/--n_values 1 2 5 10 50 100/' "${QUICK_SCRIPT}"
        sed -i 's/--x_step 0.1/--x_step 0.2/' "${QUICK_SCRIPT}"
        
        sbatch "${QUICK_SCRIPT}"
        echo "[INFO] Quick test job submitted!"
        ;;
    3)
        echo ""
        read -p "Enter n values (space-separated) [default: 1 2 3 4 5 6 7 8 9 10 20 50 100]: " N_VALUES
        N_VALUES=${N_VALUES:-"1 2 3 4 5 6 7 8 9 10 20 50 100"}
        
        read -p "Enter x step size [default: 0.1]: " X_STEP
        X_STEP=${X_STEP:-0.1}
        
        read -p "Enter CPU count [default: 32]: " CPU_COUNT
        CPU_COUNT=${CPU_COUNT:-32}
        
        read -p "Enter time limit in hours [default: 72]: " TIME_HOURS
        TIME_HOURS=${TIME_HOURS:-72}
        
        echo ""
        echo "[INFO] Creating custom job script..."
        CUSTOM_SCRIPT="${SCRIPT_DIR}/run_cubiclon_l16_custom.sh"
        cp "${JOB_SCRIPT}" "${CUSTOM_SCRIPT}"
        
        sed -i "s/#SBATCH --job-name=CubicLOn_L16_parallel/#SBATCH --job-name=CubicLOn_L16_custom/" "${CUSTOM_SCRIPT}"
        sed -i "s/#SBATCH --cpus-per-task=32/#SBATCH --cpus-per-task=${CPU_COUNT}/" "${CUSTOM_SCRIPT}"
        sed -i "s/#SBATCH --time=72:00:00/#SBATCH --time=${TIME_HOURS}:00:00/" "${CUSTOM_SCRIPT}"
        sed -i "s/--n_values 1 2 3 4 5 6 7 8 9 10 20 50 100/--n_values ${N_VALUES}/" "${CUSTOM_SCRIPT}"
        sed -i "s/--x_step 0.1/--x_step ${X_STEP}/" "${CUSTOM_SCRIPT}"
        sed -i "s/NUM_WORKERS=32/NUM_WORKERS=${CPU_COUNT}/" "${CUSTOM_SCRIPT}"
        
        sbatch "${CUSTOM_SCRIPT}"
        echo "[INFO] Custom job submitted!"
        ;;
    4)
        echo ""
        echo "Your active jobs:"
        squeue -u $(whoami)
        echo ""
        echo "Recent job history:"
        sacct -u $(whoami) --format=JobID,JobName,State,Start,End --tail=10
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "[ERROR] Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================================================="
