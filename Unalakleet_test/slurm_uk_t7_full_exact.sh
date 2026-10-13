#!/bin/bash
# =============================================================================
# Unalakleet Test 7: full  |  mode=serial  |  method=exact
# All refinements combined — canonical Unalakleet mesh
# =============================================================================
#SBATCH --job-name=uk_t7_s_e
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/uk_t7_serial_exact_%j.out
#SBATCH --error=logs/uk_t7_serial_exact_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1

module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "ocsmesh_mpi_test"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TMPDIR="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test7_full/serial/exact/tmp"
export OCSMESH_SHARED_TMPDIR="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test7_full/serial/exact/tmp"

mkdir -p "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test7_full/serial/exact" "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test7_full/serial/exact/tmp" "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/logs"

cd "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test"

echo "============================================================="
echo "  Unalakleet Test 7: full  method=exact"
echo "  Job    : ${SLURM_JOB_ID}"
echo "  Node   : ${SLURM_NODELIST}"
echo "  Date   : $(date)"
echo "  Output : /work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test7_full/serial/exact"
echo "============================================================="

python build_unalakleet_mesh.py --test 7 --mode serial --method exact --results-root '/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results'

echo "============================================================="
echo "  DONE: Test 7 / serial / exact  $(date)"
echo "============================================================="
