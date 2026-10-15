"""NJ_test/generate_slurm_scripts.py

Generate one SLURM script per (test, mode, method) combination.
Produces 60 scripts: 6 tests × 5 modes × 2 methods.
Written to slurm_gaea/ so the Hercules scripts are never overwritten.

Usage
-----
    python generate_slurm_scripts_gaea.py
    python generate_slurm_scripts_gaea.py --submit
    python generate_slurm_scripts_gaea.py --test 0 --submit
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
    MPI_MULTINODE_NODES, MPI_MULTINODE_TASKS_PER_NODE,
    PARALLEL_NPROCS, RESULTS_ROOT,
)

_THIS_DIR    = Path(__file__).resolve().parent
SCRIPT_DIR   = _THIS_DIR
SLURM_DIR    = _THIS_DIR / "slurm_gaea"       # keeps Hercules scripts untouched
LOG_DIR      = _THIS_DIR / "logs_gaea"
MANIFEST     = _THIS_DIR / "NJ_dem_manifest.json"
BUILD_SCRIPT = "build_nj_mesh_gaea.py"
CONDA_ENV    = "ocsmesh_mpi_test"
WALLTIME     = "16:00:00"

MODES = ["serial", "parallel", "mpi_no_pool", "mpi_hybrid", "mpi_multinode"]

# ---------------------------------------------------------------------------
# SLURM resource allocation per mode
# ---------------------------------------------------------------------------
_SLURM_RESOURCES = {
    "serial":        {"nodes": 1, "ntasks": 1,                 "cpus": 1,
                      "per_node": None},
    "parallel":      {"nodes": 1, "ntasks": 1,                 "cpus": PARALLEL_NPROCS,
                      "per_node": None},
    "mpi_no_pool":   {"nodes": 1, "ntasks": MPI_NOPOOL_NTASKS, "cpus": MPI_NOPOOL_CPUS,
                      "per_node": None},
    "mpi_hybrid":    {"nodes": 1, "ntasks": MPI_HYBRID_NTASKS, "cpus": MPI_HYBRID_CPUS,
                      "per_node": None},
    "mpi_multinode": {"nodes": MPI_MULTINODE_NODES, "ntasks": MPI_NOPOOL_NTASKS,
                      "cpus": MPI_NOPOOL_CPUS,
                      "per_node": MPI_MULTINODE_TASKS_PER_NODE},
}


def generate_script(
    test_id: int,
    test_name: str,
    mode: str,
    method: str,
) -> Path:
    """Generate one SLURM script and return its path."""

    job_name = f"nj_t{test_id}_{mode}_{method[:1]}"
    walltime = WALLTIME
    res      = _SLURM_RESOURCES[mode]
    nodes    = res["nodes"]
    ntasks   = res["ntasks"]
    cpus     = res["cpus"]
    per_node = res["per_node"]

    out_dir = (
        f"{RESULTS_ROOT}/test{test_id}_{test_name}/{mode}/{method}"
    )
    tmpdir  = f"{out_dir}/tmp"

    # Build the Python command arguments (same for all modes)
    py_args = (
        f"{BUILD_SCRIPT} "
        f"--test {test_id} "
        f"--mode {mode} "
        f"--method {method} "
        f"--manifest '{MANIFEST}' "
        f"--results-root '{RESULTS_ROOT}'"
    )

    # --cpus-per-task is repeated on srun: Slurm >= 22.05 no longer
    # inherits it from sbatch, which would pin the step to 1 core.
    srun_line = (
        f"srun \\\n"
        f"    --nodes={nodes} \\\n"
        f"    --ntasks={ntasks} \\\n"
        f"    --cpus-per-task={cpus} \\\n"
        f"    --cpu-bind=cores \\\n"
        f"    python {py_args}"
    )

    description = next(
        t["description"] for t in TESTS if t["id"] == test_id)

    per_node_line = (
        f"#SBATCH --ntasks-per-node={per_node}\n" if per_node else ""
    )

    content = f"""#!/bin/bash
# =============================================================================
# NJ Mesh Test {test_id}: {test_name}  |  mode={mode}  |  method={method}
# {description}
# =============================================================================
#SBATCH --job-name={job_name}
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
{per_node_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --exclusive
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --time={walltime}
#SBATCH --output={LOG_DIR}/nj_t{test_id}_{mode}_{method}_%j.out
#SBATCH --error={LOG_DIR}/nj_t{test_id}_{mode}_{method}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={SLURM_EMAIL}

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

mkdir -p "{out_dir}" "{tmpdir}" "{LOG_DIR}"

# Change to script directory so relative imports work
cd "{SCRIPT_DIR}"

echo "============================================================="
echo "  NJ Test {test_id}: {test_name}  mode={mode}  method={method}"
echo "  Job    : ${{SLURM_JOB_ID}}"
echo "  Nodes  : ${{SLURM_JOB_NUM_NODES}} (${{SLURM_NODELIST}})"
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
        f"slurm_gaea_nj_t{test_id}_{test_name}_{mode}_{method}.sh"
    )
    script_path = SLURM_DIR / script_name
    script_path.write_text(content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (and optionally submit) all 60 Gaea NJ SLURM scripts."
    )
    parser.add_argument("--submit",  action="store_true",
                        help="Submit all generated scripts with sbatch.")
    parser.add_argument("--test",    type=int,   default=None,
                        help="Generate only for this test ID.")
    parser.add_argument("--mode",    default=None,
                        choices=MODES,
                        help="Generate only for this mode.")
    parser.add_argument("--method",  default=None,
                        choices=["fast","exact"],
                        help="Generate only for this method.")
    args = parser.parse_args()

    SLURM_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    tests   = [t for t in TESTS
               if args.test is None or t["id"] == args.test]
    modes   = [m for m in MODES
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

    print(f"\n{len(scripts)} scripts generated in {SLURM_DIR}")

    if args.submit:
        print("\nSubmitting ...")
        job_ids = []
        for s in scripts:
            result = subprocess.run(
                ["sbatch", str(s)],
                capture_output=True, text=True,
                cwd=str(SLURM_DIR),
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
        print("  python generate_slurm_scripts_gaea.py --submit")
        print("\nTo submit a subset:")
        print("  python generate_slurm_scripts_gaea.py --test 0 --submit")
        print("  python generate_slurm_scripts_gaea.py --mode serial --submit")
        print(
            "  python generate_slurm_scripts_gaea.py "
            "--test 0 --mode serial --method exact --submit"
        )


if __name__ == "__main__":
    main()
