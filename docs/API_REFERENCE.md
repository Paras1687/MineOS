# MineOS AI API Reference

The backend exposes a REST API powered by FastAPI.

## Core Endpoints

| Method | Endpoint | Purpose | Input | Output |
| ------ | -------- | ------- | ----- | ------ |
| `GET` | `/api/v1/health` | Check system status | None | `{"status": "ready", ...}` |
| `GET` | `/api/v1/overview` | Fetch dashboard metrics | None | High-level metrics and alerts |

## Exploration & Geology

| Method | Endpoint | Purpose | Input | Output |
| ------ | -------- | ------- | ----- | ------ |
| `GET` | `/api/v1/mines` | List registered mines | None | Array of mine objects |
| `GET` | `/api/v1/exploration/patches` | List evaluated raster patches | None | Audit data and patch coordinates |
| `POST`| `/api/v1/exploration/predict` | Run CNN on specific coordinates | JSON (`latitude`, `longitude`, `cloud`) | Job ID (202 Accepted) |
| `GET` | `/api/v1/jobs/{uid}` | Check prediction job status | Job UUID | Job status and results if complete |
| `GET` | `/api/v1/mines/{uid}/geology` | Get geological details for a mine | Mine ID | Grade estimate and geology data |

## Operations & Production

| Method | Endpoint | Purpose | Input | Output |
| ------ | -------- | ------- | ----- | ------ |
| `GET` | `/api/v1/operational/equipment` | Get fleet health | Optional `mine_id`, `near_km` | Array of machine statuses |
| `GET` | `/api/v1/weather/{uid}` | Fetch weather forecast | Mine ID | Weather data (from Open-Meteo) |
| `POST`| `/api/v1/production/{uid}` | Predict shift production | JSON (scenario options) | Production forecast and actuals |
| `POST`| `/api/v1/decisions/{uid}` | Get ranked recovery actions | JSON (scenario options) | Ranked list of actions |
| `POST`| `/api/v1/actions/{uid}` | Record operator decision | JSON (state, reason, actor, etc.) | Confirmation status |

## Data Entry

| Method | Endpoint | Purpose | Input | Output |
| ------ | -------- | ------- | ----- | ------ |
| `POST`| `/api/v1/field-results` | Record assay results | JSON (site_id, assay_pct, notes) | Status confirmation |
| `POST`| `/api/v1/site-inputs/{uid}` | Save site configuration | JSON (costs, headroom, stockpile) | Status confirmation |
| `POST`| `/api/v1/shift-logs/{uid}` | Enter actual shift data | JSON (actual_mt, target_mt, etc.) | Status confirmation |

*Note: All POST inputs are strictly validated against Pydantic models defined in `backend/server.py`.*
