# Setup and Execution

## Prerequisites

*   Python 3.12 (as specified in the README)
*   Windows PowerShell (for executing the `.ps1` scripts)

## Installation

The repository contains setup scripts for creating the environment. 

1. Ensure Python 3.12 is installed and accessible.
2. Run the provided setup script from the root directory:
```powershell
./setup.ps1
```
This will install dependencies listed in `requirements.txt`.

## Environment Variables

*   `MINEOS_RASTERS`: (Optional) Specifies the directory containing the local TIFF files. If not set, the app expects `D:/sih data`.

Example:
```powershell
$env:MINEOS_RASTERS = 'C:/path/to/rasters'
```

## Running the Application

To start the backend server and serve the frontend:

```powershell
./run.ps1
```
Alternatively, double-click `Start-MineOS.cmd`.

The application will be available at `http://127.0.0.1:8000`.
API Documentation is available at `http://127.0.0.1:8000/docs`.

## Retraining Models (Optional)

Trained models are already included in the `backend/artifacts/` directory. However, to reproduce the models from the data:

```powershell
# Train the CNN (Requires raster dataset)
python scripts/train_cnn.py

# Train the tabular models (Production, Health, Grade)
python scripts/train_tabular.py
```

## Testing

To run the integration tests:

```powershell
python -m unittest discover -s tests -v
```
