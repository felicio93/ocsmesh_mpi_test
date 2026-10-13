#!/bin/bash
# =============================================================================
# Unalakleet Test 3: inner_depth  |  mode=serial  |  method=exact
# All above + inner depth bands (USGS DEM only, within highres_domain)
# =============================================================================
#SBATCH --job-name=uk_t3_s_e
#SBATCH --account=nos-surge
#SBATCH --partition=hercules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/uk_t3_serial_exact_%j.out
#SBATCH --error=logs/uk_t3_serial_exact_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "/work2/noaa/nos-surge/felicioc/envs/miniconda3/etc/profile.d/conda.sh"
conda activate "ocsmesh_mpi_test"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TMPDIR="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test3_inner_depth/serial/exact/tmp"
export OCSMESH_SHARED_TMPDIR="/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test3_inner_depth/serial/exact/tmp"

mkdir -p "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test3_inner_depth/serial/exact" "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test3_inner_depth/serial/exact/tmp" "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/logs"

cd "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test"

echo "============================================================="
echo "  Unalakleet Test 3: inner_depth  method=exact"
echo "  Job    : ${SLURM_JOB_ID}"
echo "  Node   : ${SLURM_NODELIST}"
echo "  Date   : $(date)"
echo "  Output : /work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results/test3_inner_depth/serial/exact"
echo "============================================================="

python build_unalakleet_mesh.py --test 3 --mode serial --method exact --results-root '/work2/noaa/nos-surge/felicioc/OCSMesh_MPI/ocsmesh_mpi_test/Unalakleet_test/results'

echo "============================================================="
echo "  DONE: Test 3 / serial / exact  $(date)"
echo "============================================================="
