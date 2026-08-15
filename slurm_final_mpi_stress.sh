#!/bin/bash
# =============================================================================
# FINAL benchmark — MPI STRESS TEST (push the implementation to its limits)
# =============================================================================
# Goal: run the MPI implementation on the LARGEST workload it can handle —
# full refinement recipe (contours + channels + all per-tile + box) and as
# many DEM tiles as fit within the 8h wall clock ACROSS MANY NODES.
#
# Unlike the other final jobs, this one does NOT need to match a serial run
# (serial could never finish this workload in 8h). It is purely a
# "how far can MPI scale" demonstration + profiling run.
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │ SIZING IS A PLACEHOLDER — DO NOT SUBMIT UNTIL THE SMOKE RUN COMPLETES.  │
# │                                                                          │
# │ After the single-node MPI smoke test, we will know:                     │
# │   - MPI wall time per tile for the meshdata dispatch stage              │
# │   - how much of the runtime is rank-0-only _apply_features (NOT sped    │
# │     up by MPI) vs MPI-distributed _calculate_and_write                  │
# │                                                                          │
# │ Use those numbers to choose:                                            │
# │   STRESS_N_CUDEM  : how many CUDEM tiles (up to the full ~370)          │
# │   SBATCH --nodes  : enough ranks that meshdata dispatch is not the      │
# │                     bottleneck; _apply_features (rank 0) will dominate  │
# │   LIGHT_FEATURES=0: full recipe                                         │
# └────────────────────────────────────────────────────────────────────────┘
#
# Submit (only after sizing): sbatch slurm_final_mpi_stress.sh
# =============================================================================

#SBATCH --job-name=ocsmesh_final_mpi_stress
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=8                       # PLACEHOLDER — resize after smoke run
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_mpi_stress_%j.out
#SBATCH --error=logs/final_mpi_stress_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail
source "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/final_config.sh"

# ── Stress-test workload (override the shared config) ───────────────────────
# Full recipe: force LIGHT_FEATURES off regardless of the environment.
LIGHT_FEATURES=0
LIGHT_FLAG=""

# Number of CUDEM tiles for the stress test. PLACEHOLDER: set after smoke run.
# Use the FULL manifest if you want the entire domain, or a large trimmed one.
STRESS_N_CUDEM="${STRESS_N_CUDEM:-100}"

# Build the stress manifest from the full manifest if it does not exist.
STRESS_MANIFEST="${SCRIPT_DIR}/dem_manifest_stress.json"
FULL_MANIFEST="${SCRIPT_DIR}/dem_manifest.json"   # the 388-tile production manifest

RESULTS_DIR="${PROJ}/results/final_mpi_stress_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}" logs

load_ocsmesh_env

if [ ! -f "${STRESS_MANIFEST}" ]; then
    if [ ! -f "${FULL_MANIFEST}" ]; then
        echo "ERROR: full manifest ${FULL_MANIFEST} not found."
        echo "  Run download_dems.py without --only to fetch the full domain,"
        echo "  or point FULL_MANIFEST at an existing large manifest."
        exit 1
    fi
    echo "--- Building stress manifest (${STRESS_N_CUDEM} CUDEM tiles) ---"
    srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/trim_manifest.py" \
        --in  "${FULL_MANIFEST}" \
        --out "${STRESS_MANIFEST}" \
        --n-cudem "${STRESS_N_CUDEM}"
fi
MANIFEST="${STRESS_MANIFEST}"

TOTAL_RANKS=$((SLURM_NNODES * 80))
NWORKERS=$((TOTAL_RANKS - 1))

print_final_config
echo "   STRESS mode   : full recipe, ${STRESS_N_CUDEM} CUDEM tiles"
echo "   Total ranks   : ${TOTAL_RANKS}  (1 manager + ${NWORKERS} workers)"

# Shared FS for cross-node intermediate files.
export TMPDIR="${RESULTS_DIR}/mpi_tmp"
mkdir -p "${TMPDIR}"

echo ""
echo "--- FINAL MPI STRESS: ${TOTAL_RANKS} ranks / ${SLURM_NNODES} nodes / full recipe ---"
srun --mpi=pmi2 \
     --ntasks=${TOTAL_RANKS} \
     --nodes=${SLURM_NNODES} \
     --ntasks-per-node=80 \
     --distribution=cyclic \
     python "${SCRIPT_DIR}/run_benchmark.py" \
        --manifest  "${MANIFEST}" \
        --shapefile "${STOFS_SHAPEFILE}" \
        --out-dir   "${RESULTS_DIR}" \
        --nprocs    "${NWORKERS}" \
        --hmin      "${HMIN}" \
        --hmax      "${HMAX}" \
        --modes     mpi

echo ""
echo "--- Generating report (mpi stress) ---"
srun --mpi=pmi2 -n 1 python "${SCRIPT_DIR}/analyze_profile.py" \
    --results-dir "${RESULTS_DIR}" \
    --out         "${RESULTS_DIR}/report_mpi_stress.txt"

echo ""
echo "================================================================="
echo " MPI STRESS complete. Results: ${RESULTS_DIR}"
echo "================================================================="
