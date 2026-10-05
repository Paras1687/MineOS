import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import ARTIFACTS
from backend.national_imagery import fetch_patch, DATE_RANGE
from backend.models.geo_intelligence.fusion_inputs import read, VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default="D:/Sample MOIL final SIH'26/India_existing_mines.csv")
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=Warning, module='rasterio')
    source = Path(args.csv)
    data = pd.read_csv(source)
    coords = np.deg2rad(data[['latitude', 'longitude']].values)
    tree = BallTree(coords, metric='haversine')
    parent = list(range(len(data)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, near in enumerate(tree.query_radius(coords, r=1.28/6371.0088)):
        for j in near:
            parent[root(int(j))] = root(i)
    groups = {}
    for i in range(len(data)):
        groups.setdefault(root(i), []).append(i)
    rows = []
    for group in groups.values():
        row = data.iloc[group[0]]
        point = dict(latitude=float(row.latitude), longitude=float(row.longitude))
        region = str(row.state)
        anchor = str(int(row.fid))
        rows.append(dict(**point, region=region, anchor=anchor, label=1, label_status='supplied_occurrence_unverified',
                         source_rows=[int(data.iloc[i].fid) for i in group], id=f'occurrence_{anchor}'))
        for angle in [45, 135, 225, 315]:
            a, b, bearing, arc = np.deg2rad(point['latitude']), np.deg2rad(point['longitude']), np.deg2rad(angle), 12/6371.0088
            dest_lat = np.arcsin(np.sin(a)*np.cos(arc)+np.cos(a)*np.sin(arc)*np.cos(bearing))
            dest_lon = b+np.arctan2(np.sin(bearing)*np.sin(arc)*np.cos(a), np.cos(arc)-np.sin(a)*np.sin(dest_lat))
            lon, lat = np.rad2deg([dest_lon, dest_lat])
            if tree.query(np.deg2rad([[lat, lon]]), k=1)[0][0, 0]*6371.0088 >= 3:
                rows.append(dict(latitude=float(lat), longitude=float(lon), region=region, anchor=anchor,
                                 label=0, label_status='sampled_unknown_background_not_absence',
                                 id=f'background_{anchor}'))
                break
    out = ARTIFACTS/'national_training'
    out.mkdir(exist_ok=True)
    (out/'requested.json').write_text(json.dumps(rows, indent=2))
    accepted, rejected = [], []
    def fetch(row):
        p, meta = fetch_patch(row['longitude'], row['latitude'])
        read(p)
        return dict(**row, path=str(p), imagery=meta)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, row): row for row in rows}
        for future in as_completed(pending):
            row = pending[future]
            try:
                accepted.append(future.result())
                print('OK', row['id'], row['region'], f'{len(accepted)}/{len(rows)}', flush=True)
            except Exception as exc:
                rejected.append(dict(**row, error=str(exc)))
                print('FAIL', row['id'], str(exc)[:140], flush=True)
            (out/'manifest.json').write_text(json.dumps(sorted(accepted, key=lambda r: r['id']), indent=2))
            (out/'rejected.json').write_text(json.dumps(rejected, indent=2))
    accepted.sort(key=lambda r: r['id'])
    if accepted:
        np.save(out/'images.npy', np.stack([read(r['path']) for r in accepted]))
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  original_rows=len(data), unique_location_groups=len(groups), requested=len(rows),
                  accepted=len(accepted), rejected=len(rejected), preprocessing=VERSION, date_range=DATE_RANGE,
                  regions={s: {str(c): sum(r['region']==s and r['label']==c for r in accepted) for c in [0, 1]}
                           for s in sorted(set(r['region'] for r in rows))},
                  label_scope='Supplied occurrence versus sampled unlabelled background. Source provenance is unverified; no reserve/absence inference.',
                  legacy_positive_india_excluded='Unaligned 10m/20m bands, divided index channels and missing raster georeferencing in legacy downloader')
    (out/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
