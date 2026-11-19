# AK_test: Alaska mesh across the antimeridian

Reference mesh for the OCSMesh antimeridian (0-360) work. The domain crosses
180 in the Bering Sea and is meshed in 0-360 longitudes (0 = Greenwich,
180 = Bering Sea, 360 = Greenwich).

1. **Reference:** today's OCSMesh (v3 `main`) plus the Alaska workarounds,
   branch `ak/lon360-patched`:
   - `RegionConstraint` wraps points to 0-360 (fix 2);
   - the `fast` composite wraps to 0-360 (fix 3);
   - `OCSMESH_METRIC_CRS` forces the metric CRS, EPSG:3338 (fix 4). This
     replaces the old script-level monkeypatch, which worker processes did
     not see.

   Fix 1 (GeometryCollection into Gmsh) is in v3 already. `main` also has
   `bnd_factor`, which the October runs did not have.
2. **Comparison:** the same recipe on the OCSMesh branch with antimeridian
   support, run with `WORKAROUNDS=0`. The two meshes should match.

## Files
| File | Purpose |
|---|---|
| `build_ak_mesh.py` | Shifts the inputs to 0-360, builds Geom + Hfun, runs `MeshDriver`, wraps the output to 0-360, writes `.2dm`/`.gr3` and a summary JSON |
| `slurm_ak_mesh.sh` | Gaea C6 batch script |

## Case `m5_no_center` (default)
`m4_plus_coarse` (the October mesh) without the center region:

| refinement | kind | value | rate |
|---|---|---|---|
| `shelf_150` | patch | 1500 m | 1.48e-5 |
| `below_150_patch` | patch | 2500 m | none |
| `below_200` | region, min | 2500 m | 5e-5 |
| `bering_2500` | patch | 2500 m | none |
| `coarse_south`, `coarse_north` | region, min | 3500 m | 1e-5 |

hmin 1500 m, hmax 3500 m, Gmsh `adapt` boundary with `bnd_factor` 1.5.
Cases `m1`-`m4` are still available through `CASE`.

## Inputs
`DATA_DIR` (default `.../AK_LargeDomain_Mesh/2026_10_15`) holds the original
-180..180 files: `AKdomain_20261014.shp`, `150m_contour.shp`,
`below_150m_contour.shp`, `below_200m_contour.shp`, `beringsea.shp`,
`coarse_ocean_{south,north}.shp`, `GEBCO_2026.tif`. They are shifted to 0-360
at run time. GEBCO is rolled once into `GEBCO_2026_360.tif`, which is reused
while it is newer than the original.

## Run
```bash
sbatch slurm_ak_mesh.sh                                   # reference
sbatch --export=ALL,METHOD=fast slurm_ak_mesh.sh          # fast method
sbatch --export=ALL,AK_OCSMESH=<checkout>,WORKAROUNDS=0 slurm_ak_mesh.sh
```
Outputs: `.../AK_LargeDomain_Mesh/results/<case>_<method>_bnd<f>_w<0|1>_<jobid>/`.
The log and `*_summary.json` record the OCSMesh path and commit actually
imported; check them before comparing runs.
