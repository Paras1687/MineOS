from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'backend' / 'data'
ARTIFACTS = ROOT / 'backend' / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RASTERS = Path(os.getenv('MINEOS_RASTERS', 'D:/sih data'))
MASTER = DATA / 'Manganese_Master_Dataset.csv'
DB = DATA / 'mineos.sqlite3'
