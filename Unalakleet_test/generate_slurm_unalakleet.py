"""Unalakleet_test/generate_slurm_unalakleet.py

Generate SLURM scripts for Unalakleet benchmark.
8 tests × 1 mode (mpi_no_pool) × 2 methods = 16 runs.

Usage
-----
    python generate_slurm_unalakleet.py           # generate only
    python generate_slurm_unalakleet.py --submit  # generate + submit
    python generate_slurm_unalakleet.py --test 0 --submit
"""

from __future__ import annotations

import argparse
import stat
import subprocess
from pathlib import Path

from unalakleet_config import (
    TESTS, METHODS,
    SLURM_ACCOUNT, SLURM_PARTITION, SLURM_EMAIL, SLURM_WALLTIME,
    MPI_NOPOOL_NTASKS, MPI_NOPOOL_CPUS,
    RESULTS_ROOT,
)

_THIS_DIR  = Path(__file__).resolve().parent
PROJ       = "/work2/noaa/nos-surge/felicioc/OCSMesh_MPI"
CONDA_BASE = "/work2/noaa/nos-surge/felicioc/envs/miniconda3"
CONDA_ENV  = "ocsmesh_mpi_test"
SCRIPT_DIR = str(_THIS_DIR)


def generate_script(
    test_id: int,
    test_name: str,
    method: str,
) -> Path:
    """Generate one SLURM script for mpi_no_pool + given method."""

    mode     = "mpi_no_pool"
    job_name = f"uk_t{test_id}_{method[:1]}"
    out_dir  = f"{RESULTS_ROOT}/test{test_id}_{test_name}/{mode}/{method}"
    tmpdir   = f"{out_dir}/tmp"

    py_args = (
        f"build_unalakleet_mesh.py "
        f"--test {test_id} "
        f"--mode {mode} "
        f"--method {method} "
        f"--results-root '{RESULTS_ROOT}'"
    )

    srun_line = (
        f"srun --mpi=pmi2 \\\n"
        f"    --ntasks={MPI_NOPOOL_NTASKS} \\\n"
        f"    --cpus-per-task={MPI_NOPOOL_CPUS} \\\n"
        f"    python {py_args}"
    )

    description = next(
        t["description"] for t in TESTS if t["id"] == test_id)

    content = f"""#!/bin/bash
# =============================================================================
# Unalakleet Test {test_id}: {test_name}  |  mode={mode}  |  method={method}
# {description}
# =============================================================================
#SBATCH --job-name={job_name}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks={MPI_NOPOOL_NTASKS}
#SBATCH --cpus-per-task={MPI_NOPOOL_CPUS}
#SBATCH --exclusive
#SBATCH --time={SLURM_WALLTIME}
#SBATCH --output=logs/uk_t{test_id}_{method}_%j.out
#SBATCH --error=logs/uk_t{test_id}_{method}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM_EMAIL}

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel-oneapi-compilers/2022.2.1
module load intel-oneapi-mpi/2021.7.1
module load hdf5/1.12.2
module load netcdf-c/4.9.0
module load netcdf-fortran/4.6.0
source "{CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "{CONDA_ENV}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TMPDIR="{tmpdir}"
export OCSMESH_SHARED_TMPDIR="{tmpdir}"

mkdir -p "{out_dir}" "{tmpdir}" "{_THIS_DIR}/logs"

cd "{SCRIPT_DIR}"

echo "============================================================="
echo "  Unalakleet Test {test_id}: {test_name}  method={method}"
echo "  Job    : ${{SLURM_JOB_ID}}"
echo "  Node   : ${{SLURM_NODELIST}}"
echo "  Date   : $(date)"
echo "  Output : {out_dir}"
echo "============================================================="

srun --mpi=pmi2 -n 1 python -c \\
    "from mpi4py import MPI; print('mpi4py OK:', MPI.Get_version())"

{srun_line}

echo "============================================================="
echo "  DONE: Test {test_id} / {mode} / {method}  $(date)"
echo "============================================================="
"""

    script_name = f"slurm_uk_t{test_id}_{test_name}_{method}.sh"
    script_path = _THIS_DIR / script_name
    script_path.write_text(content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Unalakleet SLURM scripts (mpi_no_pool only)."
    )
    parser.add_argument("--submit",  action="store_true")
    parser.add_argument("--test",    type=int,  default=None)
    parser.add_argument("--method",  default=None, choices=["fast", "exact"])
    args = parser.parse_args()

    (_THIS_DIR / "logs").mkdir(exist_ok=True)

    tests   = [t for t in TESTS
               if args.test is None or t["id"] == args.test]
    methods = [m for m in METHODS
               if args.method is None or m == args.method]

    scripts = []
    for t in tests:
        for method in methods:
            path = generate_script(t["id"], t["name"], method)
            scripts.append(path)
            print(f"  Generated: {path.name}")

    print(f"\n{len(scripts)} scripts generated in {_THIS_DIR}")

    if args.submit:
        print("\nSubmitting ...")
        for s in scripts:
            result = subprocess.run(
                ["sbatch", str(s)],
                capture_output=True, text=True,
                cwd=str(_THIS_DIR),
            )
            if result.returncode == 0:
                jid = result.stdout.strip().split()[-1]
                print(f"  Submitted {s.name} → job {jid}")
            else:
                print(f"  FAILED {s.name}: {result.stderr.strip()}")
    else:
        print("\nTo submit all:")
        print("  python generate_slurm_unalakleet.py --submit")
        print("To submit a subset:")
        print("  python generate_slurm_unalakleet.py --test 0 --submit")
        print("  python generate_slurm_unalakleet.py --method fast --submit")


if __name__ == "__main__":
    main()
