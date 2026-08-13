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
salloc -N 1 -n 2 -t 00:10:00 -A nos-surge -p hercules

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

## #4 — Confirmed working: `srun --mpi=pmi2` on Hercules

**Symptom / status**
Resolution of #3. With mpi4py built from source against Intel MPI (#2),
this launches correctly:

```bash
srun --mpi=pmi2 -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank', c.Get_rank(), 'of', c.Get_size())"
# rank 1 of 2
# rank 0 of 2
```

**Takeaway**
`--mpi=pmi2` is the confirmed working PMI on Hercules. Both benchmark
SLURM scripts now pass `--mpi=pmi2` to every `srun` MPI launch. If a
future Hercules change breaks this, run `srun --mpi=list` and switch to
the supported type (e.g. `pmix`).

**Confirmed Slurm/allocation values (from a working nos-surge job card):**
- `--account=nos-surge`
- `--partition=hercules`
- Node size: **80 cores / 512 GB per node** (use `--exclusive`)
- Windfall queue note: max 450 nodes, no walltime limit but lower priority;
  for quick turnaround keep walltime <= 8 hrs.

**OCSMesh-side?**
No — site config. Recorded here so the SLURM scripts and the OCSMesh MPI
docs can cite a known-good Hercules launch recipe.

---

## #5 — ETOPO2022 THREDDS WCS returns HTTP 400 (no WCS for that dataset)

**Symptom**
`download_dems.py` GEBCO step fails on every attempt:
```
400 Client Error: 400 for url:
https://www.ngdc.noaa.gov/thredds/wcs/global/etopo2022/ETOPO_2022_v1_15s_N90W180_bed.nc?...WCS...
WARNING: GEBCO download failed.
```

**Root cause**
The NCEI THREDDS server does NOT expose a WCS service for the ETOPO2022
global dataset (and the single global `N90W180` NetCDF the URL assumed
does not exist there). The dataset is served as **per-tile 15° NetCDF
files** via fileServer / OPeNDAP / NCSS only — not WCS. So the WCS
`GetCoverage` request is invalid and returns 400.

Confirmed catalog layout:
```
/thredds/catalog/global/ETOPO2022/15s/15s_bed_elev_netcdf/catalog.html
  -> ETOPO_2022_v1_15s_N60W105_bed.nc, ...N60W090..., etc. (15°x15° tiles)
Plain-file download base:
  /thredds/fileServer/global/ETOPO2022/15s/15s_bed_elev_netcdf/<tile>.nc
```

**Workaround (recommended)**
Provide the deep-ocean background GeoTIFF yourself — no WCS needed:

1. Download a GEBCO grid for the domain from https://download.gebco.net/
   (GeoTIFF), e.g. `gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif`.
2. Drop it in `$DEMS/gebco/` OR set `GEBCO_LOCAL` to its full path:
   ```bash
   export GEBCO_LOCAL=$DEMS/gebco/gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif
   ```
`download_dems.py` now (a) honors `GEBCO_LOCAL`, (b) auto-detects any
`*.tif` already sitting in `$DEMS/gebco/`, and only then (c) falls back
to downloading + merging the ETOPO2022 15" NetCDF tiles with rasterio.

**OCSMesh-side?**
No — this is a data-sourcing issue in the benchmark's download helper,
now fixed. Not an OCSMesh code concern.

---

## #6 — CUDEM tile downloads 404 (hardcoded filenames were stale/guessed)

**Symptom**
Many CUDEM tiles 404 during download, e.g.:
```
404 Client Error: Not Found for url:
https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/FL/ncei19_n25x00_w082x00_2020v1.tif
```
Some subfolders (MA_NH_ME, rima, chesapeake_bay, NC) downloaded fine;
others (southeast, FL, LA_MS, TX) had many misses.

**Root cause**
`download_dems.py` originally carried HARDCODED tile-name lists. For the
subfolders that were scraped completely up-front they were correct; for
the larger ones (FL ~109 tiles, LA_MS ~119, TX ~97, southeast ~75) the
lists were partially hand-guessed, so the filenames (wrong year/version
suffix, wrong lon/lat step) did not exist on S3 → 404.

**Fix**
Scrape each subfolder's `index.html` at RUNTIME and regex out the real
`ncei19_*.tif` names, then pick every other tile:
```
https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/<subfolder>/index.html
```
Implemented as `get_subfolder_tiles(subfolder)` with a compiled regex
`ncei19_[ns]\\d+[xX]\\d+_[ew]\\d+[xX]\\d+_\\d{4}v\\d+\\.tif` (uppercase X
allowed — some FL tiles use it). No filenames are hardcoded anymore, so
the lists can never go stale.

**OCSMesh-side?**
No — benchmark data-sourcing only. Not an OCSMesh code concern.

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
