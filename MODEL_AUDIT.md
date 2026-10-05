# India CNN audit and tuning — 5 October 2026

## Result

The newly trained national candidate did not pass deployment criteria. It is saved for reproducibility but is not the active Exploration model. Reliable nationwide manganese occurrence, grade or reserve prediction is not established.

| Evaluation | ROC-AUC | PR-AUC |
|---|---:|---:|
| New CNN, nested regional out-of-fold predictions | 0.437642 | 0.445177 |
| Spectral statistics + gravity baseline, same outer regions | 0.431973 | 0.452239 |
| Final CNN ensemble, training set only | 0.620748 | 0.639198 |

Mean regional CNN ROC-AUC: 0.457463. These metrics distinguish supplied occurrences from sampled unknown backgrounds; backgrounds are not independently verified absence. Training-set performance is not validation.

## Correction to the earlier 0.994 claim

The earlier India experiment initialized every fold from a global checkpoint whose training included the same occurrence sites. Its positives were also concentrated in the mining belt while the selected barren controls represented desert terrain. A latitude/longitude-only classifier scored ROC-AUC 1.00 using the same grouped folds. Therefore the reported 0.993537 is retained only as a confounded legacy diagnostic, not independent India accuracy. Active metadata now explicitly records these limitations. The unsafe legacy trainer is retired; its original source is preserved outside the application in `work/model_audit/legacy_train_india_fusion.py`.

## Data prepared

- Source: `D:/Sample MOIL final SIH'26/India_existing_mines.csv`, unchanged; 54 supplied records across seven source-labelled regions.
- Duplicate/overlapping sites within 1.28 km were grouped, leaving 47 location groups.
- Requested one occurrence patch and one sampled background 12 km away per group: 94 patches. Backgrounds remain labelled unknown, not confirmed barren. Their primarily single-bearing sampling is a limitation.
- Downloaded 92 usable cloud-masked, aligned Sentinel-2 L2A patches from the 2025 reference window. Both Ladakh patches failed quality checks and were excluded.
- Four pairs with acquisition dates more than 60 days apart were excluded to reduce seasonal confounding.
- Evaluated 84 patches: 42 supplied occurrences and 42 unknown backgrounds across MP-labelled mine belt, Odisha, Andhra, Karnataka, Gujarat and Rajasthan. Some regions have only two occurrences.
- All six reflectance bands are reprojected to the same 20 m grid. NDVI, NDWI and NDMI are recomputed from aligned reflectances. Each input covers 1.28 km at 64×64 pixels. Measured Bouguer-grid values supply the gravity input.
- The old `Positive_India` TIFFs were excluded because their downloader padded native 10 m and 20 m bands without spatial alignment and divided index channels again.

## Training and validation

A compact CNN with gravity fusion, GroupNorm, dropout, AdamW, gradient clipping, orientation augmentation and small physically consistent reflectance-gain augmentation was trained. Two learning-rate/regularization settings were compared inside each outer training partition. Inner regional validation chose settings and early stopping; three independently initialized models were averaged. No pretrained weights were used in validation.

Every outer fold held out a complete supplied region, with a 5 km exclusion buffer and training-only normalization. The observed minimum outer train/test distance was 93.467 km. Same-site samples remained together. Split IDs, source hashes, scene dates, predictions and settings are saved. No score thresholds or known-site scores were manually increased.

## Application changes and checks

- Fixed the active regional model's live footprint mismatch: training used 640 m at 10 m/pixel, while live input had covered 1.28 km. Point requests and nearby scan now use the correct regional-model scale; map footprints match it.
- Live FID-1 cloud check: 21.816667 N, 80.166667 E; 29 September 2026 Sentinel scene, valid fraction 1.0, 640 m footprint, model score 87.6798/100. This is an uncalibrated regional screening score, not deposit probability or independent validation.
- Added versioned aligned imagery caching and a national input pipeline. National weights can load only after the deployment gate passes and checkpoint checksum matches. The current candidate failed that gate.
- Reports now distinguish the active model from the national candidate and display the candidate's actual regional results instead of another CNN's metrics.
- 25 automated tests passed, including reflectance units, index mathematics, physical footprint equivalence, missing-image rejection, normalization isolation, deployment gating and existing app workflows. JavaScript syntax check passed.

The active model remains `moil-india-fusion-adaptation-v1`, explicitly marked regional and unvalidated nationally. The 0.994 diagnostic is no longer presented as independent validation.

## Files and reproduction

- `scripts/prepare_national_fusion.py`: source audit, grouping and reference imagery retrieval.
- `scripts/train_national_fusion.py`: nested regional training, baseline comparison and deployment gate.
- `backend/national_imagery.py` and `backend/models/geo_intelligence/fusion_inputs.py`: shared input contract and cache.
- `backend/models/geo_intelligence/national_fusion.py`: compact multimodal CNN and ensemble.
- `backend/models/geo_intelligence/world_fusion.py`, `backend/cloud.py`, `backend/exploration.py`: model selection, scale corrections and point/scan integration.
- `backend/server.py`, `frontend/js/portal.js`: accurate model evidence and report display.
- `backend/artifacts/national_training/`: source audit, manifests, exclusions, arrays and out-of-fold predictions.
- `backend/artifacts/national_fusion_report.json` and `national_fusion.pt`: measured results and inactive candidate weights.

Run preparation and training offline from the application root using its Python environment. Model training is never triggered by clicking the map.

## Still needed for a defensible national model

Independent, provenance-checked occurrence coordinates and comparable verified negative/assay sites from several geological belts; geology and subsurface evidence that are available at prediction time; additional seasons and an untouched external regional test. Augmentation cannot supply this missing evidence. Nine satellite bands and coarse gravity alone do not establish mineable reserves.
