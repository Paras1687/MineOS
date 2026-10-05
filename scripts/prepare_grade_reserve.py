"""Offline preparation and spatial evaluation. Never invoked by point requests."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from backend.config import ARTIFACTS
from backend.grade_data import FEATURES, normalize, parse_grade, parse_reserve, km

METHOD_VERSION = 'grade-reserve-v1'


def regressor():
    return Pipeline([('encode', ColumnTransformer([('geo', OneHotEncoder(handle_unknown='ignore'), FEATURES)])),
                     ('model', RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                                                     max_depth=6, random_state=42, n_jobs=2))])


def prepare(source):
    raw = source.read_bytes()
    frame = pd.read_csv(source).fillna('')
    rows = []
    invalid = []
    for i, row in frame.iterrows():
        try:
            lat, lon = float(row.Latitude), float(row.Longitude)
            assert np.isfinite([lat, lon]).all() and -90 <= lat <= 90 and -180 <= lon <= 180
        except (ValueError, AssertionError):
            invalid.append(i+2); continue
        rows.append(dict(row=i+2, name=str(row.Mine_Name), latitude=lat, longitude=lon,
                         region='India' if 'India' in str(row.Source) else 'International',
                         source=str(row.Source), geology={f:normalize(row[f]) for f in FEATURES},
                         grade=parse_grade(row.Grade), reserve=parse_reserve(row.Reserve),
                         raw={c:str(row[c]) for c in frame.columns}))
    # Connected location groups within 20 m, not random row splitting.
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    pairs = []
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]['region'] == rows[j]['region'] and km(rows[i], rows[j]) <= .02:
                parents[root(i)] = root(j); pairs.append([rows[j]['row'], rows[i]['row']])
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(root(i), []).append(row)
    groups = []
    for n, members in enumerate(grouped.values()):
        labels = [r['grade']['midpoint_pct'] for r in members
                  if r['grade']['midpoint_pct'] is not None and not r['grade'].get('quality_flags')]
        selected = max(members, key=lambda r:sum(v != 'unknown' for v in r['geology'].values()))
        groups.append(dict(id=f'LOC-{n:04d}', latitude=float(np.mean([r['latitude'] for r in members])),
                           longitude=float(np.mean([r['longitude'] for r in members])),
                           region=selected['region'], geology=selected['geology'], records=members,
                           target=float(np.median(labels)) if labels else None))
    models, evaluation = {}, {}
    for region in ['India', 'International']:
        items = [g for g in groups if g['region'] == region and g['target'] is not None]
        # Leave a 0.5-degree spatial cell out, then purge training locations within 10 km.
        cells = sorted(set((int(np.floor(g['latitude']*2)), int(np.floor(g['longitude']*2))) for g in items))
        folds, errors, base_errors = [], [], []
        for cell in cells:
            test = [g for g in items if (int(np.floor(g['latitude']*2)), int(np.floor(g['longitude']*2))) == cell]
            train = [g for g in items if g not in test and min(km(g,t) for t in test) >= 10]
            if len(train) < 8:
                folds.append(dict(cell=cell, status='insufficient training locations', test_ids=[g['id'] for g in test]));continue
            model = regressor().fit(pd.DataFrame([g['geology'] for g in train]), [g['target'] for g in train])
            y = np.array([g['target'] for g in test]);pred = model.predict(pd.DataFrame([g['geology'] for g in test]))
            baseline = float(np.median([g['target'] for g in train]))
            errors.extend(np.abs(y-pred).tolist());base_errors.extend(np.abs(y-baseline).tolist())
            folds.append(dict(cell=cell,status='evaluated',train_ids=[g['id'] for g in train],
                              test_ids=[g['id'] for g in test], min_distance_km=min(km(a,b) for a in train for b in test),
                              actual_midpoints=y.tolist(),predicted=pred.tolist(),baseline=baseline))
        report = dict(training_location_groups=len(items), evaluated_groups=len(errors), folds=folds,
                      mae_pct=float(np.mean(errors)) if errors else None,
                      median_baseline_mae_pct=float(np.mean(base_errors)) if base_errors else None,
                      screening_error_p90_pct=float(np.quantile(errors,.9)) if errors else None)
        report['beats_baseline'] = bool(errors and report['mae_pct'] < report['median_baseline_mae_pct'])
        report['prediction_enabled'] = report['beats_baseline'] and len(errors) >= 15
        evaluation[region] = report
        if len(items) >= 8:
            models[region] = regressor().fit(pd.DataFrame([g['geology'] for g in items]), [g['target'] for g in items])
    source_hash = hashlib.sha256(raw).hexdigest()
    model_path = ARTIFACTS/'grade_spatial_v1.joblib'
    joblib.dump(models, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    version = METHOD_VERSION+'-'+source_hash[:12]+'-'+model_hash[:12]
    audit = dict(rows=len(frame), valid_coordinate_rows=len(rows), invalid_coordinate_csv_rows=invalid,
                 region_rows=dict(Counter(r['region'] for r in rows)),
                 region_location_groups=dict(Counter(g['region'] for g in groups)), duplicate_pairs=pairs,
                 grade_kinds=dict(Counter(r['grade']['kind'] for r in rows)),
                 flagged_grade_rows=[r['row'] for r in rows if r['grade'].get('quality_flags')],
                 reserve_units=dict(Counter((r['reserve']['original_unit'] or 'unspecified') if r['reserve'] else 'missing' for r in rows)),
                 repeated_descriptions={c:frame[c].value_counts().head(5).to_dict() for c in ['Drill_Depth','Ore_Thickness','Subsurface_Grade']})
    metadata = dict(version=version, data_sha256=source_hash, model_sha256=model_hash,
                    source_path=str(source), source_filename=source.name, features=FEATURES,
                    audit=audit, evaluation=evaluation,
                    method='Region-specific random forest geological analogue; 0.5-degree held-out cells with 10 km training purge.',
                    label_basis='Exact Mn values and midpoint of finite Mn ranges. Duplicate-location targets use median of available midpoint labels. Bounds/missing and >65% review labels excluded from fitting.',
                    limitations=['Source provenance unverified; this measures agreement with supplied midpoint labels, not assays.',
                                 'P90 residual range is descriptive cross-validation error, not a calibrated confidence interval.',
                                 'Positive occurrence dataset only: grade conditional on mineralization, not proof of occurrence.',
                                 'Repeated generic geometry descriptions cannot support site tonnage.'])
    (ARTIFACTS/'grade_reserve_data.json').write_text(json.dumps(dict(version=version,groups=groups),indent=2),encoding='utf-8')
    (ARTIFACTS/'grade_reserve_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({**audit,'evaluation':{k:{a:b for a,b in v.items() if a!='folds'} for k,v in evaluation.items()},'version':version},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);args=parser.parse_args();prepare(args.source)
