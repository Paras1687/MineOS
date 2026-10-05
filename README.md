# MineOS AI — integrated advisory prototype

Built on the supplied project. Original E: project and D: imagery are unchanged.

## Run

On this computer, the prepared workspace uses `../mineos-env/Scripts/python.exe`.
Double-click `Start-MineOS.cmd` (or run `./run.ps1` in PowerShell), then open http://127.0.0.1:8000.
The server stays local to this computer. API documentation: http://127.0.0.1:8000/docs.

For a fresh extracted package, install Python 3.12, run `./setup.ps1` once (internet required), then `./run.ps1`.
Trained models are included; no retraining is required to run the app. The large TIFF collection is not duplicated.
Local inference expects the supplied TIFF folders under `D:/sih data` (`Positive` and `Negative_True_Barren`). `Negative_Unknowns` are excluded from the CNN.
If relocated, set `$env:MINEOS_RASTERS = 'your raster folder'` before launching.
Weather and cloud imagery require internet; unavailable inputs are labelled rather than replaced by fake live data.

## Working features

- Nine-band CNN training, grouped spatial holdout, separate calibration and TIFF quality checks. Point inspection displays a 0–100 CNN score after the fitted sample calibrator; it is not a manganese occurrence probability. When the inspected TIFF has a supplied label, the UI explicitly flags disagreement as a model error. The point report also shows training-patch counts within 25 km. Offline point inference refuses training/calibration tiles and permits only held-out test tiles; automatic comparative site ranking is currently disabled until India-specific positive holdout patches and independent field validation are available.
- Public Sentinel-2 L2A cloud retrieval, cloud masking, date/source disclosure and Grad-CAM evidence.
- Exploration screening order, known-label/split visibility, nearby occurrence records and conditional geology-based grade estimates.
- Mine-specific fleet maps, sensor-based failure classification, equipment type/status/standby filters and machine detail.
- Shift/day/week/month conditional production scenarios combining equipment, weather and editable blasting delays.
- Ranked stockpile, transfer and schedule alternatives with costs, capacity/travel constraints, recovery and net benefit.
- Persistent operator decisions, exclusive alternative approval, machine reservations, actual outcome records and field assay intake.
- Evidence dashboard with measured evaluation results and explicit source limitations.

## What the evidence supports

The active app CNN was retrained from `D:/sih data` using 449 accepted positive and 1,455 user-designated barren patches (1,904 accepted of 1,967 selected). Global spatial test ROC-AUC is 0.915 and PR-AUC is 0.853 on 264 patches. The India test subset has 16 patches but zero positives, so this score is not evidence of India accuracy. Barren labels are not independently verified by drilling or assays. Treat displayed values as experimental indices, not occurrence probabilities; automatic ranking remains disabled. Full audit: `backend/experiments/sih_barren_2026_09_28/experiment_summary.json`.

## What the evidence supports

| Component | Measured result | Interpretation |
|---|---|---|
| Imagery audit | 1,887 TIFFs; 1,825 accepted; 242 spatial groups | Six exact duplicates and 56 insufficient-coverage patches excluded |
| CNN | Global spatial holdout ROC-AUC 0.915; PR-AUC 0.853; prevalence 0.284; 264 patches | Experimental only; test geography is imbalanced and India test has 0 positive patches. User-designated barren terrain is not assay-verified. No India accuracy claim; automatic comparative ranking disabled. |
| XGBoost benchmark | Held-out ROC-AUC 0.607; PR-AUC 0.354; same 306-patch global test split | Candidate uses nine-band per-patch distribution summaries and was selected on validation only. Paired spatial-group bootstrap AUC difference vs CNN: 95% interval −0.033 to 0.204; India test subset has zero patches. Experimental only; not promoted to the India model |
| Random Forest benchmark | Held-out ROC-AUC 0.574; PR-AUC 0.339; same 306-patch global test split | Same 64 image-summary features. Paired spatial-group bootstrap AUC difference vs CNN: 95% interval −0.055 to 0.171; India test subset has zero patches. Does not beat XGBoost on this split; experimental only |
| Gaussian Naive Bayes benchmark | Held-out ROC-AUC 0.519; PR-AUC 0.255; same 306-patch global test split | Same 64 image-summary features. Paired spatial-group bootstrap AUC difference vs CNN: 95% interval −0.077 to 0.064; India test subset has zero patches. Not suitable as the exploration model |
| Logistic Regression benchmark | Held-out ROC-AUC 0.502; PR-AUC 0.262; same 306-patch global test split | Standardized 64 image-summary features; C selected on validation. Paired spatial-group bootstrap AUC difference vs CNN: 95% interval −0.106 to 0.059; India test subset has zero patches. Not suitable as the exploration model |
| Equipment | Test PR-AUC 0.746; recall 0.750; precision 0.593 | Synthetic sensor benchmark failure-state classification; not future failure timing or remaining useful life |
| Production | Shift MAE 28.50 MT vs rolling baseline 43.98 MT; 1,323 test rows | Conditional retrospective test with realized weather, not validation of future weather-driven forecasts |
| Grade | Held-out midpoint MAE 3.65 percentage points | Experimental analogue model using supplied text labels; no independently verified assay validation |

The new local CNN bundle is deployed in the packaged app. Live cloud retrieval was previously verified for 21.816667 N, 80.166667 E. The returned scene was acquired 21 June 2026; it is **not a live satellite observation**. Open-Meteo forecast retrieval was also verified. All eight integration tests passed.

## Data limitations that remain

- Machine coordinates, standby status, capacities and mine assignments are simulated. The benchmark contains no real GPS stream.
- Three supplied production blocks are reused as labelled demo templates for 25 distinct India sites (five near-duplicate CSV locations merged). Aggregate totals are demo scenarios, not MOIL forecasts. Current actual production is unavailable.
- Resource/tonnage prediction is deliberately unavailable: mixed/unknown reserve units, lack of verified depth/volume/density and source provenance prevent defensible reserve estimation.
- Arbitrary map points have no verified geology. Grade is only offered at supported known occurrences and remains experimental.
- Blasting, prices, operating costs, plant headroom and stockpile quantities are editable assumptions.
- Model scores are calibrated to the sampled dataset; they are not the probability of an economically mineable deposit. Background samples are not verified sterile drilling locations.
- Maintenance triage is available; repair diagnosis, reliable repair-price prediction, live dispatch, ERP integration and causal recovery validation need real operating data.
- Operator approval records an advisory decision. It does not physically move equipment or automatically alter schedules.
- This local prototype has no authentication or production deployment hardening.

## Reproduce

From the project root with the environment's Python:

```powershell
python scripts/train_cnn.py --help
python scripts/train_tabular.py
python scripts/rank_prospects.py
python -m unittest discover -s tests -v
```

CNN training creates `backend/artifacts/patches.npy`, omitted from the distribution because it is a regenerable 269 MB training cache. Ranking regeneration requires this cache. Test metrics, model weights, manifest and rejected-file audit are included. Existing benchmark training CSVs remain under backend/data and backend/models/production_forecast.

The active entry point is `backend.server:app`, with `frontend/js/portal.js` and `frontend/css/portal.css`. Older source modules in the development copy are not served by this entry point.

## Second-round demo

1. Overview: disclose real imagery/forecast feeds and simulated operations.
2. Exploration: inspect a point and show acquisition date, score, Grad-CAM and the weak-model quality gate.
3. Fleet: select a mine and inspect equipment health, location provenance and standby candidates.
4. Production: fetch weather; compare a delay scenario across periods.
5. Decision centre: show an action's recovery, travel/capacity limits and cost calculation.
6. Evidence: explain held-out results and the specific field/operating data needed to improve them.

Do not claim validated reserves, high exploration accuracy, live equipment GPS, guaranteed savings or superiority over another team. Stronger CNN validation needs independently verified labels, geological covariates and spatial/external evaluation, not a more favourable random split.

## Third-party data and UI

Sentinel-2 imagery: Copernicus / Microsoft Planetary Computer. Weather: Open-Meteo (CC BY 4.0). Map tiles: OpenStreetMap contributors. Leaflet 1.9.4 is distributed under BSD-2-Clause (vendor/LICENSE-Leaflet.txt). User-supplied datasets retain their source terms and require provenance verification before redistribution outside the team.
