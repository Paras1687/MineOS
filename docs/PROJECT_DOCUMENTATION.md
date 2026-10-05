# 1. Project Overview

**Project Name:** MineOS AI
**One-Line Description:** An integrated advisory prototype for manganese exploration, production forecasting, and operational decision-support.
**Problem Addressed:** The project addresses the problem of manganese reserve identification, production shortfall forecasting, and operational risk analysis through AI/ML-driven decision support.
**Why it Matters:** Mining operations face uncertainties from geological anomalies, weather, equipment downtime, and varying production rates. Predictive analytics can optimize operational decisions to mitigate shortfall risks.
**Target Users:** Mining planners, exploration teams, operations managers, and decision-makers.
**Core Objective:** Provide evidence-based AI screening for exploration, production forecasting, and operational fleet health based on verifiable input data.

**Executive Summary:** MineOS AI is a prototype web application designed for the SIH problem statement on manganese reserves and production shortfall. The system implements a multimodal Convolutional Neural Network (CNN) to screen Sentinel-2 imagery for manganese prospectivity, an XGBoost model for production forecasting based on weather and equipment availability, and a Random Forest classification model for synthetic machine health. It features an integrated dashboard displaying geospatial exploration, fleet conditions, and production scenarios, allowing operators to review predictive recommendations and explicitly record actual outcomes. The codebase focuses heavily on highlighting data limitations, particularly the lack of verified subsurface geometry and tonnages.

# 2. Problem Statement

Mining operations currently struggle with disconnected data sources—geological exploration, fleet maintenance, and daily shift production are often siloed. Traditional approaches to forecasting lack the ability to integrate live weather, operational standby data, and predictive machine failure risks simultaneously. As a result, planners are often reactive to shortfalls rather than proactive. This system aims to support decisions regarding where to direct exploration efforts (via CNN screening) and how to manage fleet allocation during potential production delays (via predictive modelling).

# 3. Proposed Solution

The system provides an integrated pipeline for operations:
Data Sources (Images, CSVs, APIs)
→ Data Collection (Planetary Computer Sentinel-2, Open-Meteo, AI4I sensors, production history)
→ Preprocessing (Patch extraction, cloud masking, standard scaling, lag features)
→ Machine Learning Models (CNN, XGBoost, Random Forest)
→ Prediction (Prospectivity screening, shift production MT, machine failure probability, grade classification)
→ Decision Support (Alternative ranking, cost-benefit calculation)
→ Dashboard (Interactive Leaflet maps, Plotly charts, UI forms for operator overrides)

# 4. Key Features

| Feature | Description | Status |
| ------- | ----------- | ------ |
| **Exploration CNN** | 9-band Sentinel-2 imagery & gravity anomaly screening for prospectivity index | Implemented |
| **Predictive Maintenance** | Sensor-based classification of machine failure probability | Implemented |
| **Production Forecasting** | Conditional shift output prediction incorporating weather and lag features | Implemented |
| **Geological Grade Estimation**| RF model predicting grade mid-point based on categorical geological labels | Implemented |
| **Recovery Decision Center**| Ranks stockpile and transfer alternatives based on costs and capacity constraints | Implemented |
| **Resource/Tonnage Estimation**| Estimation of absolute metric reserve volume and density | Proposed (Currently blocked due to lack of verified depth/volume data) |

# 5. System Architecture

The architecture consists of a FastAPI backend serving a static HTML/JS frontend, powered by PyTorch and scikit-learn/XGBoost models.

```
User
 ↓ (Browser / HTML / JS)
Frontend (Leaflet maps, API calls)
 ↓ (HTTP / REST)
Backend API (FastAPI)
 ↓
Data Processing & Model Inference 
  ├─ Geo-Intelligence (PyTorch CNN, Planetary Computer API)
  ├─ Production (XGBoost, Open-Meteo API, Pandas)
  └─ Maintenance & Grade (Scikit-Learn, Random Forest)
 ↓
Database / Artifact Store (SQLite, JSON, Joblib, PT)
```

# 6. Repository Structure

```text
MineOS-AI/
├── backend/
│   ├── models/                # ML model definitions (geo_intelligence, production_forecast)
│   ├── data/                  # CSV datasets (Manganese_Master, AI4I, etc.)
│   ├── artifacts/             # Trained weights (.pt, .joblib) and inference caches
│   ├── server.py              # FastAPI application entry point
│   ├── store.py               # SQLite database setup and query logic
│   ├── config.py              # Path routing and configuration
│   └── weather.py             # Open-Meteo API integration
├── frontend/
│   ├── js/                    # Client-side logic (portal.js)
│   ├── css/                   # Stylesheets (portal.css)
│   ├── vendor/                # Third-party libraries (Leaflet, Plotly)
│   └── *.html                 # Dashboard views
├── scripts/                   # Model training and data prep scripts
└── tests/                     # Unit and integration tests
```

# 7. Dataset and Data Sources

1. **Sentinel-2 L2A Imagery**: Fetched via Copernicus / Microsoft Planetary Computer (9 spectral bands used).
2. **AI4I 2020 Predictive Maintenance**: Synthetic benchmark sensor data (`ai4i2020_train_FINAL.csv`). Contains air temp, process temp, rotational speed, torque, tool wear, and machine failure flag.
3. **Manganese Master Dataset**: Custom CSV containing geological records (Geology, Lithology, Age, Grade) and coordinates.
4. **Production History**: Operator-entered/simulated shift production (`MineOS_Production_Intelligence_Train.csv`).

*Note: The project explicitly states that equipment coordinates and benchmark sensors are synthetic mapping and not live GPS streams.*

# 8. Feature Engineering

*   **Production Lags**: `lag1` (previous shift production) and `rolling7` (7-shift rolling average). Useful for capturing recent momentum.
*   **Weather**: `Rainfall_mm`, `Temperature_C`, `Humidity_pct`. Represents environmental conditions impacting operations.
*   **Equipment Aggregates**: `Predicted_Overall_Equipment_Availability_pct` and `Predicted_Average_Equipment_Failure_Risk_pct`.
*   **Geological Encodings**: `Geology`, `Lithology`, `Host_Rock`, `Formation`, `Age` are One-Hot Encoded to predict the `grade_mid`.
*   **Raster Summaries**: The CNN evaluates 9-band spatial patches (including pixel valid fractions and gravity grids).

# 9. Machine Learning Methodology

*   **Exploration (NationalFusionCNN)**: PyTorch CNN. Binary classification task (prospectivity index 0-100). Uses Sentinel-2 patches + gravity anomalies. Trained using BCEWithLogitsLoss with spatial nested leave-one-region-out cross-validation.
*   **Predictive Maintenance (RandomForestClassifier)**: Scikit-learn. Predicts binary `Machine failure` from 5 sensor columns.
*   **Production Forecasting (XGBRegressor)**: XGBoost. Predicts continuous `Actual_Production_MT`. Trained on historical shift records with conformal prediction used for calculating error bounds (radius).
*   **Grade Estimation (RandomForestRegressor)**: Scikit-learn. Predicts continuous `grade_mid` based on geological text labels.

# 10. Model Evaluation

Evaluation is thoroughly documented via artifacts (`backend/artifacts/*.json`):

*   **CNN Exploration**: Global spatial test ROC-AUC 0.915, PR-AUC 0.853 (Note: India test subset had 0 positive patches in the split, indicating lack of India-specific performance).
*   **Equipment (Health)**: Test PR-AUC 0.746, Recall 0.750, Precision 0.593.
*   **Production**: Shift MAE 28.50 MT vs rolling baseline 43.98 MT.
*   **Grade**: Held-out midpoint MAE 3.65 percentage points.

# 11. Prediction / Forecasting Workflow

1.  **Input**: User selects a site and shift/date via the Production UI.
2.  **Processing**: Backend fetches live weather (Open-Meteo) and applies operator overrides (e.g., blast delays).
3.  **Feature Generation**: `lag1` and `rolling7` are computed from the historical database (`production_history.csv`). Fleet risk is aggregated.
4.  **Inference**: XGBoost predicts `Actual_Production_MT`.
5.  **Post-Processing**: Shortfall is calculated against the `target_per_shift_mt`.
6.  **Output**: Data is sent to the frontend for charting (Plotly/SVG).

# 12. Risk Analysis / Decision Support

*   **Operational Risk**: The system calculates the gross revenue at risk due to predicted shortfalls.
*   **Decision Center**: If a shortfall is predicted, the backend (`backend.planning.alternatives`) ranks "recovery actions" (e.g., processing stockpiles, transferring material).
*   **Logic**: Actions are scored on `net_benefit_inr` = Gross value - variable cost - action cost, bounded by transfer constraints (distance/speed) and processing capacity.

# 13. Dashboard / User Interface

*   **Dashboard**: High-level metrics and site locations on a Leaflet map.
*   **Exploration**: Allows clicking arbitrary coordinates. Triggers a live fetch of Sentinel-2 imagery, runs the CNN, and displays a Grad-CAM heatmap overlay.
*   **Maintenance**: Tabular view of machines, showing model-predicted risk scores and standby eligibility.
*   **Production**: Forms to override weather/blasting, displaying SVG charts of historical vs predicted output.
*   **Decision Center**: Review and explicitly "Acknowledge", "Accept", or "Reject" the proposed recovery alternative.

# 14. Backend and API Documentation

See `docs/API_REFERENCE.md` for full details. The backend is built on FastAPI and exposes RESTful endpoints taking JSON payloads (via Pydantic models).

# 15. Frontend Architecture

The frontend is a vanilla HTML/CSS/JS single-page-like application (though split across multiple HTML files).
*   **Framework**: Vanilla JS (`frontend/js/portal.js`).
*   **Mapping**: Leaflet.js (`frontend/vendor/leaflet.js`).
*   **State**: Simple global `state` object holding selected mines and UI status.
*   **API Comms**: Standard `fetch` wrapped in an `api()` helper function.

# 16. End-to-End Workflow

1.  **Discover**: User visits Exploration, clicks a coordinate in India. The backend fetches Sentinel-2 data, runs the CNN, and returns a prospectivity score (e.g., 92/100).
2.  **Plan**: User goes to Production, selects a mine. The system fetches live weather, predicts a shortfall of 500 MT for Shift-1 due to heavy rain.
3.  **Decide**: User moves to Decision Center. The system proposes moving 500 MT from Stockpile A to cover the gap.
4.  **Act**: User clicks "Approve planned recovery", logging the decision in the database.

# 17. Installation and Setup

See `docs/SETUP.md`.

# 18. Configuration and Environment Variables

Required environment configuration is minimal for local testing.
*   `MINEOS_RASTERS`: (Optional) Path to locally cached raster files if not located at the default `D:/sih data`.

# 19. Running the Project

See `docs/SETUP.md`.

# 20. Deployment

No production deployment configuration (e.g., Dockerfile, Gunicorn config, or cloud infrastructure) is included in the current prototype. The backend runs locally via Uvicorn.
**Future Scope:** Containerize via Docker and deploy on a managed service like AWS ECS or Azure Container Apps with a managed PostgreSQL instance replacing SQLite.

# 21. Limitations

*   **Data Provenance**: Benchmark failure data is synthetic (AI4I).
*   **Geographic Validity**: The CNN's India test split had 0 positive patches, meaning India-specific accuracy is unproven.
*   **Missing Features**: Tonnage/Reserve estimation is deliberately blocked in code due to lack of 3D depth/volume data.
*   **Live Data**: Machine GPS locations and status are simulated.

# 22. Security and Reliability

*   **Security**: No authentication/authorization is implemented. SQLite database is unencrypted.
*   **Validation**: Pydantic models strictly validate API inputs (e.g., `latitude` between -90 and 90, `value` >= 0).
*   **Reliability**: The backend handles API timeouts gracefully and implements rate limiting (max 4 concurrent CNN inferences) via a threading lock.

# 23. Testing

The repository contains tests in `tests/`:
*   `test_cloud.py`
*   `test_estimation.py`
*   `test_national_fusion.py`
*   `test_nearby.py`
*   `test_system.py`
The README notes "All eight integration tests passed."

# 24. Results

Observed results inside the active prototype:
*   CNN Global ROC-AUC: 0.915 (experimental screening index).
*   Equipment benchmark PR-AUC: 0.746.
*   Production Shift MAE: 28.50 MT.
*   UI successfully renders live Grad-CAM imagery dynamically.

# 25. Business / Real-World Impact

For operations managers, this prototype proves the concept of integrating disparate mining data. Instead of making separate phone calls to maintenance, geology, and weather forecasters, a shift planner can immediately see that a storm is approaching, two excavators are high-risk, and the resulting shortfall can be economically offset by drawing from a nearby stockpile.

# 26. Future Scope

*   **Verified Geological Data**: Incorporate actual borehole logs and assay data to enable genuine 3D resource/reserve tonnage estimations.
*   **Live Telemetry**: Replace AI4I synthetic data with actual SCADA/IoT feeds from mining equipment.
*   **Cloud Deployment**: Secure the API with OAuth2 and deploy to a cloud environment.

# 27. Conclusion

MineOS AI is a highly mature prototype that successfully demonstrates a full end-to-end data pipeline from satellite inference to shift-level financial decision support. While heavily limited by the quality and availability of real-world mining data (which the codebase transparently acknowledges and handles gracefully), the underlying AI architecture and software engineering are robust and clearly address the SIH problem statement's core requirements.

# Implementation Evidence

| Claim | Evidence in Repository |
| ----- | ---------------------- |
| XGBoost Production Model | `backend/models/production_forecast/` and `scripts/train_tabular.py` |
| PyTorch CNN Exploration | `backend/models/geo_intelligence/national_fusion.py` |
| AI4I Synthetic Benchmark | `backend/data/ai4i2020_train_FINAL.csv` |
| Pydantic Input Validation | `backend/server.py` (`Options`, `Point`, `Decision` classes) |
| Missing Tonnage Warning | `scripts/train_tabular.py` (line 67: "Blocked: unit and provenance verification required") |
| API Rate Limiting | `backend/server.py` (`point_prediction` max 4 concurrent jobs) |
