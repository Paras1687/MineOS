# Render exploration repair

Inspected repository commit: bbd738c. Checked 6 October 2026.

## Observed

- Deployed health, mines, Exploration HTML, Leaflet and Plotly returned HTTP 200.
- A deployed prediction at 21.816667 N, 80.166667 E started, then job polling returned HTTP 502. The service subsequently responded again. Render process logs were unavailable, so an out-of-memory termination is suspected, not confirmed.
- Satellite reprojection passed an entire remote raster band to GDAL. It now reads only a padded source window for the requested footprint.
- Gravity checksum allocated the complete 233,410,500-byte file. It now hashes in 1 MB blocks and retains the memory-mapped file handle.
- New imagery is cached on disk by coordinates, resolution, size and quality threshold, with a checksum and source metadata. This survives application restarts while the disk remains; Render ephemeral storage can be cleared on redeploy.
- Catalog DNS/network/429/5xx errors retry. Failure of the second catalog search no longer discards successful first-search scenes.
- Matching prediction jobs are reused; point and nearby jobs remain distinct. Failed jobs can be retried.
- Frontend stops polling obsolete selections, prevents duplicate scan clicks, tolerates temporary gateway errors, and explains expired jobs.
- Build preparation downloads and verifies the Git LFS gravity asset if missing or unresolved. Model weights and training data are unchanged.

## Verification

Actual cloud fetch at the location above: 64 x 64 x 9 patch, 100% clear pixels, scene S2B_MSIL2A_20260929T050649_R019_T44QMK_20260929T091314, 11.6 seconds on this workstation.

Complete inference using that imagery: score 0.8769898847, gravity 72.7273788452 mGal, model moil-india-fusion-adaptation-v1. These are screening outputs, not independently validated mineral findings.

Real 5 km scan: 45 sampled patches, 43 scored, 2 missing sufficient clear imagery, 19 scores >= 0.9, 35.2 seconds on this workstation. Render latency and memory usage may differ.

Final regression run: 29/29 tests passed.

Regression coverage includes imagery quality filtering, persistent-cache reuse, DNS retry, cropped raster alignment, streaming checksum, job deduplication, grade parsing, nearby radius, production and fleet workflows. JavaScript syntax checked with node --check.

## Apply to the existing Render service

Push the repair branch and merge it after reviewing the diff. No GitHub or Render update was performed from this workspace because authenticated GitHub write access was unavailable.

For an existing manually configured Render service, set:

Build command: `pip install -r requirements.txt && python scripts/prepare_deployment.py`

Start command: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT --workers 1`

Health check: `/api/v1/health`

Environment: `GDAL_CACHEMAX=16`, `GDAL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. The included render.yaml declares these settings for Blueprint deployments; adding it alone does not reconfigure an existing manually managed service.

After redeploy, repeat point prediction and the nearby scan and inspect Render memory/restart logs. The single-process queue is intentionally in memory; deployments invalidate active jobs. Satellite outages or cloudy imagery can still prevent a supported score. No guarantee of every-point coverage is possible.
