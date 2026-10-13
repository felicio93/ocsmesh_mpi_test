#!/bin/bash
# =============================================================================
# Unalakleet Test 6: channels  |  mode=serial  |  method=exact
# All above + channel mask patch (20m inside polygon)
# =============================================================================
#SBATCH --job-name=uk_t6_s_e
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/uk_t6_serial_exact_%j.out
#SBATCH --error=logs/uk_t6_serial_exact_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge


module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "ocsmesh_mpi_test"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TMPDIR="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test6_channels/serial/exact/tmp"
export OCSMESH_SHARED_TMPDIR="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test6_channels/serial/exact/tmp"

mkdir -p "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test6_channels/serial/exact" "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test6_channels/serial/exact/tmp" "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/logs"

cd "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test"

echo "============================================================="
echo "  Unalakleet Test 6: channels  method=exact"
echo "  Job    : ${SLURM_JOB_ID}"
echo "  Node   : ${SLURM_NODELIST}"
echo "  Date   : $(date)"
echo "  Output : /work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test6_channels/serial/exact"
echo "============================================================="

python build_unalakleet_mesh.py --test 6 --mode serial --method exact --results-root '/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results'

echo "============================================================="
echo "  DONE: Test 6 / serial / exact  $(date)"
echo "============================================================="
