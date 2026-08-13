# Hercules Gotchas & OCSMesh MPI Feedback Log

Running log of issues hit while testing the OCSMesh MPI implementation on
the NOAA RDHPC (Hercules), with root cause, workaround, and whether it is
something the OCSMesh side can fix/improve.

Share the "OCSMesh-side?" column with the OCSMesh MPI developer.

Environment (as tested):
- Cluster: Hercules (Rocky 9, Slurm)
- Modules: `intel-oneapi-compilers/2022.2.1`, `intel-oneapi-mpi/2021.7.1`,
  `hdf5/1.12.2`, `netcdf-c/4.9.0`, `netcdf-fortran/4.6.0`
- Conda env: `ocsmesh_mpi_test` (Python 3.10)
- OCSMesh: editable install from `$PROJ/OCSMesh`

---

## #1 — `pip install -e .` does not install mpi4py

**Symptom**
After `pip install -e .`, `import mpi4py` fails with
`ModuleNotFoundError: No module named 'mpi4py'`, even though ocsmesh
imports fine.

**Root cause**
`mpi4py` is declared as an *optional* dependency in OCSMesh's
`pyproject.toml`:

```toml
dependencies = [ ... ]          # core deps — mpi4py NOT included
[project.optional-dependencies]
mpi = ["mpi4py"]                # only installed via the ".[mpi]" extra
```

A plain `pip install -e .` installs only core deps by design.

**Workaround**
Install mpi4py explicitly (see #2 — do NOT just add `.[mpi]` on HPC).

**OCSMesh-side?**
Docs only. The optional-extra design is intentional. Worth documenting
clearly in the OCSMesh MPI setup guide that mpi4py is not auto-installed.

---

## #2 — `.[mpi]` extra pulls a prebuilt mpi4py wheel linked to the wrong MPI

**Symptom**
`pip install -e ".[mpi]"` succeeds and `import mpi4py` works on the login
node, but under `srun` the MPI layer misbehaves / does not see the Intel
MPI ranks (silent wrong-MPI, or init failures).

**Root cause**
On PyPI, `pip` grabs a prebuilt mpi4py wheel that bundles / links a
generic MPI (often MPICH), NOT the site's Intel MPI. mpi4py must be
compiled against the exact MPI that `srun` uses.

**Workaround**
Build mpi4py from source against the loaded Intel MPI module:

```bash
module load intel-oneapi-mpi/2021.7.1
which mpicc                                   # must resolve to Intel MPI
MPICC=$(which mpicc) pip install --no-binary=mpi4py --no-cache-dir mpi4py
```

The build log should say "Building wheel for mpi4py ... done" (compiled),
not "Downloading mpi4py-...whl".

**OCSMesh-side?**
Docs mainly. Could optionally add a note / helper in the OCSMesh MPI docs
recommending the `MPICC=... --no-binary=mpi4py` pattern for HPC installs.

---

## #3 — `PMI2_Job_GetId returned 14` when verifying mpi4py

**Symptom**
```
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread:
Other MPI error, error stack:
MPIR_Init_thread(176): ...
MPIR_pmi_init(167)...: PMI2_Job_GetId returned 14
```

**Root cause**
Running an MPI program as a bare `python -c ...` on a login / interactive
shell. `MPI_Init` tries to contact a PMI (process management interface)
server that only exists when the job is launched under `srun`/`mpirun`.
No launcher → no PMI → abort at init. This is NOT an OCSMesh bug and NOT
an install problem — mpi4py is fine.

**Workaround**
Always launch MPI code through Slurm with a matching PMI:

```bash
# Inside an allocation:
salloc -N 1 -n 2 -t 00:10:00 -A <account> -p <partition>

# Verify with srun (pick the PMI your Slurm supports):
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"

# If pmi2 is not supported, list options and try another (e.g. pmix):
srun --mpi=list
```

If Intel MPI + Slurm still disagree, point Intel MPI at the Slurm PMI lib:
```bash
export I_MPI_PMI_LIBRARY=/opt/slurm/lib/libpmi2.so   # path is site-specific
```

**OCSMesh-side?**
Partially. The environment plumbing is outside OCSMesh, BUT
`MPIExecutor` could catch a failed `MPI_Init` / PMI error and emit a
friendly message such as:
> "MPI failed to initialize. Are you running under srun/mpirun with the
>  correct --mpi=pmiX? A bare 'python script.py' cannot initialize MPI."
instead of surfacing a raw `PMI2_Job_GetId` abort. Good UX improvement.

---

## Template for new entries

```
## #N — <one-line symptom>

**Symptom**
<paste exact error>

**Root cause**
<why>

**Workaround**
<commands>

**OCSMesh-side?**
<Docs only | UX improvement possible | Code fix needed | No>
```
