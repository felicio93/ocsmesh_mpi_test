"""NJ_test/generate_slurm_scripts.py

Generate one SLURM script per (test, mode, method) combination.
Produces 48 scripts: 6 tests × 4 modes × 2 methods.

Usage
-----
    python generate_slurm_scripts.py
    python generate_slurm_scripts.py --submit
    python generate_slurm_scripts.py --test 0 --submit
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
from pathlib import Path

from nj_mesh_config_gaea import (
    TESTS, METHODS,
    SLURM_ACCOUNT, SLURM_PARTITION, SLURM_EMAIL,
    MPI_NOPOOL_NTASKS, MPI_NOPOOL_CPUS,
    MPI_HYBRID_NTASKS, MPI_HYBRID_CPUS,
    PARALLEL_NPROCS,
)

_THIS_DIR    = Path(__file__).resolve().parent
SCRIPT_DIR   = _THIS_DIR
RESULTS_ROOT = _THIS_DIR / "results"
MANIFEST     = _THIS_DIR / "NJ_dem_manifest.json"
PROJ         = "/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI"
CONDA_ENV    = "ocsmesh_mpi_test"

# ---------------------------------------------------------------------------
# Walltime estimates per (test_id, mode)
# ---------------------------------------------------------------------------
_WALLTIMES = {
    (0, "serial"):       "16:00:00",
    (0, "parallel"):     "16:00:00",
    (0, "mpi_no_pool"):  "16:00:00",
    (0, "mpi_hybrid"):   "16:00:00",
    (1, "serial"):       "16:00:00",
    (1, "parallel"):     "16:00:00",
    (1, "mpi_no_pool"):  "16:00:00",
    (1, "mpi_hybrid"):   "16:00:00",
    (2, "serial"):       "16:00:00",
    (2, "parallel"):     "16:00:00",
    (2, "mpi_no_pool"):  "16:00:00",
    (2, "mpi_hybrid"):   "16:00:00",
    (3, "serial"):       "16:00:00",
    (3, "parallel"):     "16:00:00",
    (3, "mpi_no_pool"):  "16:00:00",
    (3, "mpi_hybrid"):   "16:00:00",
    (4, "serial"):       "16:00:00",
    (4, "parallel"):     "16:00:00",
    (4, "mpi_no_pool"):  "16:00:00",
    (4, "mpi_hybrid"):   "16:00:00",
    (5, "serial"):       "16:00:00",
    (5, "parallel"):     "16:00:00",
    (5, "mpi_no_pool"):  "16:00:00",
    (5, "mpi_hybrid"):   "16:00:00",
}

# ---------------------------------------------------------------------------
# SLURM resource allocation per mode
# ---------------------------------------------------------------------------
_SLURM_RESOURCES = {
    "serial":      {"ntasks": 1,                 "cpus": 1},
    "parallel":    {"ntasks": 1,                 "cpus": PARALLEL_NPROCS},
    "mpi_no_pool": {"ntasks": MPI_NOPOOL_NTASKS, "cpus": MPI_NOPOOL_CPUS},
    "mpi_hybrid":  {"ntasks": MPI_HYBRID_NTASKS, "cpus": MPI_HYBRID_CPUS},
}


def generate_script(
    test_id: int,
    test_name: str,
    mode: str,
    method: str,
) -> Path:
    """Generate one SLURM script and return its path."""

    job_name = f"nj_t{test_id}_{mode[:3]}_{method[:1]}"
    walltime = _WALLTIMES.get((test_id, mode), "16:00:00")
    res      = _SLURM_RESOURCES[mode]
    ntasks   = res["ntasks"]
    cpus     = res["cpus"]

    out_dir = (
        f"{RESULTS_ROOT}/test{test_id}_{test_name}/{mode}/{method}"
    )
    tmpdir  = f"{out_dir}/tmp"

    # Build the Python command arguments (same for all modes)
    py_args = (
        f"build_nj_mesh.py "
        f"--test {test_id} "
        f"--mode {mode} "
        f"--method {method} "
        f"--manifest '{MANIFEST}' "
        f"--results-root '{RESULTS_ROOT}'"
    )

    # Build the srun line — Python is the direct executable for all modes
    if mode in ("serial", "parallel"):
        srun_line = (
            f"srun -n 1 \\\n"
            f"    python {py_args}"
        )
    elif mode == "mpi_no_pool":
        srun_line = (
            f"srun \\\n"
            f"    --ntasks={ntasks} \\\n"
            f"    --cpus-per-task={cpus} \\\n"
            f"    python {py_args}"
        )
    elif mode == "mpi_hybrid":
        srun_line = (
            f"srun \\\n"
            f"    --ntasks={ntasks} \\\n"
            f"    --cpus-per-task={cpus} \\\n"
            f"    --overcommit \\\n"
            f"    python {py_args}"
        )

    description = next(
        t["description"] for t in TESTS if t["id"] == test_id)

    overcommit_note = (
        "# --overcommit: 6 ranks × 7 cores = 42 cores, fits on 1 node\n"
        if mode == "mpi_hybrid" else ""
    )

    content = f"""#!/bin/bash
# =============================================================================
# NJ Mesh Test {test_id}: {test_name}  |  mode={mode}  |  method={method}
# {description}
# =============================================================================
#SBATCH --job-name={job_name}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --exclusive
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --time={walltime}
#SBATCH --output=logs/nj_t{test_id}_{mode}_{method}_%j.out
#SBATCH --error=logs/nj_t{test_id}_{mode}_{method}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM_EMAIL}

{overcommit_note}
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load PrgEnv-intel
export LD_LIBRARY_PATH=${{CRAY_LD_LIBRARY_PATH}}:${{LD_LIBRARY_PATH}}
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_RX_MATCH_MODE=software
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate "{CONDA_ENV}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# TMPDIR on Lustre — avoids node-local /tmp overflow with many MPI ranks
export TMPDIR="{tmpdir}"
export OCSMESH_SHARED_TMPDIR="{tmpdir}"

mkdir -p "{out_dir}" "{tmpdir}" "{_THIS_DIR}/logs"

# Change to script directory so relative imports work
cd "{SCRIPT_DIR}"

echo "============================================================="
echo "  NJ Test {test_id}: {test_name}  mode={mode}  method={method}"
echo "  Job    : ${{SLURM_JOB_ID}}"
echo "  Node   : ${{SLURM_NODELIST}}"
echo "  Date   : $(date)"
echo "  Output : {out_dir}"
echo "============================================================="

# ── Verify mpi4py ─────────────────────────────────────────────────────────────
srun -n 1 python -c \\
    "from mpi4py import MPI; print('mpi4py OK:', MPI.Get_version())"

# ── Run ───────────────────────────────────────────────────────────────────────
{srun_line}

echo "============================================================="
echo "  DONE: Test {test_id} / {mode} / {method}"
echo "  $(date)"
echo "============================================================="
"""

    script_name = (
        f"slurm_nj_t{test_id}_{test_name}_{mode}_{method}.sh"
    )
    script_path = _THIS_DIR / script_name
    script_path.write_text(content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (and optionally submit) all 48 NJ SLURM scripts."
    )
    parser.add_argument("--submit",  action="store_true",
                        help="Submit all generated scripts with sbatch.")
    parser.add_argument("--test",    type=int,   default=None,
                        help="Generate only for this test ID.")
    parser.add_argument("--mode",    default=None,
                        choices=["serial","parallel","mpi_no_pool","mpi_hybrid"],
                        help="Generate only for this mode.")
    parser.add_argument("--method",  default=None,
                        choices=["fast","exact"],
                        help="Generate only for this method.")
    args = parser.parse_args()

    (_THIS_DIR / "logs").mkdir(exist_ok=True)

    tests   = [t for t in TESTS
               if args.test is None or t["id"] == args.test]
    modes   = [m for m in
               ["serial","parallel","mpi_no_pool","mpi_hybrid"]
               if args.mode is None or m == args.mode]
    methods = [m for m in METHODS
               if args.method is None or m == args.method]

    scripts = []
    for t in tests:
        for mode in modes:
            for method in methods:
                path = generate_script(
                    t["id"], t["name"], mode, method)
                scripts.append(path)
                print(f"  Generated: {path.name}")

    print(f"\n{len(scripts)} scripts generated in {_THIS_DIR}")

    if args.submit:
        print("\nSubmitting ...")
        job_ids = []
        for s in scripts:
            result = subprocess.run(
                ["sbatch", str(s)],
                capture_output=True, text=True,
                cwd=str(_THIS_DIR),
            )
            if result.returncode == 0:
                # Gaea sbatch output: "Submitted batch job JOBID on cluster c6"
                jid_parts = result.stdout.strip().split()
                jid = jid_parts[3] if len(jid_parts) >= 4 else jid_parts[-1]
                job_ids.append(jid)
                print(f"  Submitted {s.name} → job {jid}")
            else:
                print(f"  FAILED {s.name}: {result.stderr.strip()}")
        print(f"\n{len(job_ids)} jobs submitted.")
        print("Monitor: squeue -u $USER")
    else:
        print("\nTo submit all:")
        print("  python generate_slurm_scripts.py --submit")
        print("\nTo submit a subset:")
        print("  python generate_slurm_scripts.py --test 0 --submit")
        print("  python generate_slurm_scripts.py --mode serial --submit")
        print(
            "  python generate_slurm_scripts.py "
            "--test 0 --mode serial --method exact --submit"
        )


if __name__ == "__main__":
    main()
