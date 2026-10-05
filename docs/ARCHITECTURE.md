# MineOS AI Architecture

## System Architecture

The MineOS AI system is structured as a client-server web application.

```mermaid
flowchart TD
    Client[Browser / User Interface] -->|HTTP Requests| FastAPI[FastAPI Backend]
    
    sublayer1[Frontend Layer]
    sublayer1 -.-> Client
    
    sublayer2[Backend Layer]
    sublayer2 -.-> FastAPI
    
    FastAPI -->|Queries| SQLite[(SQLite Database)]
    FastAPI -->|Inference| MLModels[ML Artifacts & Models]
    
    MLModels --> CNN[PyTorch CNN]
    MLModels --> XGB[XGBoost Regressor]
    MLModels --> RF[Scikit-Learn RF]
    
    FastAPI -->|API Calls| ExtAPI[External APIs]
    ExtAPI --> PC[Planetary Computer]
    ExtAPI --> OM[Open-Meteo]
```

## Data Flow Pipeline

```mermaid
flowchart LR
    A[Data Sources] -->|Imagery| B(Preprocessing)
    A -->|Sensors| B
    A -->|Production| B
    B --> C{Feature Engineering}
    C -->|Rolling Averages| D[XGBoost Forecast]
    C -->|Tensor Conversion| E[PyTorch CNN]
    D --> F[Prediction Output]
    E --> F
    F --> G[Decision Center Logic]
    G --> H[UI Dashboard]
```

## ML Training Pipeline

```mermaid
flowchart TD
    Raw[Raw CSV / GeoTIFFs] --> Split[Train/Validation Split]
    Split --> Train[Model Training]
    
    Train --> CNN[train_national_fusion.py]
    Train --> Tabular[train_tabular.py]
    
    CNN --> Eval1[Spatial Holdout Evaluation]
    Tabular --> Eval2[Temporal/Random Evaluation]
    
    Eval1 --> Artifacts[Saved Artifacts .pt / .joblib]
    Eval2 --> Artifacts
    
    Artifacts --> Inference[FastAPI Server Inference]
```

## Module Responsibilities

1. **`backend/server.py`**: The core FastAPI application. Handles routing, input validation (via Pydantic), and manages a thread pool for asynchronous CNN inference.
2. **`backend/planning.py`**: Contains the logic for the Decision Center, ranking alternative recovery actions based on costs.
3. **`backend/exploration.py`**: Interacts with the CNN model and the Planetary Computer API to fetch and evaluate raster imagery.
4. **`frontend/js/portal.js`**: Drives the user interface, managing state, rendering Leaflet maps, and dynamically updating UI elements based on API responses.
