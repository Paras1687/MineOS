import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, log_loss
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import ARTIFACTS
from backend.models.geo_intelligence.national_fusion import NationalFusionCNN, augment
from backend.models.geo_intelligence.world_fusion import gravity_grid
from backend.models.geo_intelligence.fusion_inputs import VERSION

CONFIGS = [dict(name='regularized', lr=.001, decay=.05), dict(name='low_rate', lr=.0003, decay=.1)]


def probability(model, x, g, idx):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(x[idx], g[idx])).numpy()


def metrics(y, p):
    return dict(n=len(y), positives=int(sum(y)), roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p>=.5)),
                log_loss=float(log_loss(y, p, labels=[0, 1])))


def fit(x, y, g, train, val, config, seed, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NationalFusionCNN()
    model.fit_normalization(x[train], g[train])
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['decay'])
    lossfn = torch.nn.BCEWithLogitsLoss()
    best, best_state, best_epoch, stale = float('inf'), None, 1, 0
    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train)/16)))):
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(augment(x[ix], rng), g[ix]), y[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        if val is None:
            best_epoch = epoch+1
            continue
        score = log_loss(y[val].numpy(), probability(model, x, g, val), labels=[0, 1])
        if score < best-1e-4:
            best, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best


def main():
    torch.set_num_threads(4)
    folder = ARTIFACTS/'national_training'
    rows = json.loads((folder/'manifest.json').read_text())
    complete_anchors = {r['anchor'] for r in rows if r['label']==1} & {r['anchor'] for r in rows if r['label']==0}
    seasonal_exclusions = []
    for anchor in sorted(complete_anchors):
        dates = [datetime.fromisoformat(r['imagery']['acquisition_date'].replace('Z', '+00:00'))
                 for r in rows if r['anchor']==anchor]
        if (max(dates)-min(dates)).days > 60:
            seasonal_exclusions.append(anchor)
    complete_anchors -= set(seasonal_exclusions)
    keep = [i for i, r in enumerate(rows) if r['anchor'] in complete_anchors]
    images = np.load(folder/'images.npy')[keep]
    rows = [rows[i] for i in keep]
    x = torch.tensor(images)
    labels = np.array([r['label'] for r in rows])
    y = torch.tensor(labels, dtype=torch.float32)
    regions = np.array([r['region'] for r in rows])
    gravity = np.array([gravity_grid()([[r['latitude'], r['longitude']]])[0] for r in rows], dtype='float32')[:, None]
    g = torch.tensor(gravity)
    coords = np.deg2rad([[r['latitude'], r['longitude']] for r in rows])
    features = np.concatenate([images.mean((2, 3)), images.std((2, 3)), gravity], axis=1)
    oof, baseline = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    folds, selections = [], []
    states = sorted(set(regions))
    if len(states) < 4:
        raise ValueError('At least four complete regions required')
    for fold, region in enumerate(states):
        test = np.flatnonzero(regions == region)
        remainder = np.flatnonzero(regions != region)
        near = BallTree(coords[test], metric='haversine').query(coords[remainder], k=1)[0][:, 0]*6371.0088
        remainder = remainder[near >= 5]
        inner_regions = sorted(set(regions[remainder]))
        validation_region = inner_regions[fold % len(inner_regions)]
        val = remainder[regions[remainder] == validation_region]
        train = remainder[regions[remainder] != validation_region]
        gap = BallTree(coords[val], metric='haversine').query(coords[train], k=1)[0][:, 0]*6371.0088
        train = train[gap >= 5]
        for indices in [train, val, test]:
            if len(np.unique(labels[indices])) != 2:
                raise ValueError('Region split lacks both sample classes')
        candidates = []
        for config in CONFIGS:
            _, epochs, loss = fit(x, y, g, train, val, config, 700+fold)
            candidates.append(dict(config=config, epochs=epochs, validation_log_loss=loss))
        selected = min(candidates, key=lambda c: c['validation_log_loss'])
        selections.extend(candidates)
        predictions = []
        for seed in [701, 702, 703]:
            model, _, _ = fit(x, y, g, remainder, None, selected['config'], seed, selected['epochs'])
            predictions.append(probability(model, x, g, test))
        oof[test] = np.mean(predictions, axis=0)
        baseline_model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000))
        baseline_model.fit(features[remainder], labels[remainder])
        baseline[test] = baseline_model.predict_proba(features[test])[:, 1]
        result = dict(region=region, n_train=len(remainder), validation_region=validation_region,
                      train_ids=[rows[i]['id'] for i in remainder], test_ids=[rows[i]['id'] for i in test],
                      inner_train_ids=[rows[i]['id'] for i in train], validation_ids=[rows[i]['id'] for i in val],
                      min_train_test_distance_km=float(BallTree(coords[test], metric='haversine').query(coords[remainder], k=1)[0].min()*6371.0088),
                      selected=selected, cnn=metrics(labels[test], oof[test]),
                      baseline=metrics(labels[test], baseline[test]))
        folds.append(result)
        print(json.dumps({k: result[k] for k in ['region','selected','cnn','baseline']}), flush=True)
    cnn = metrics(labels, oof)
    base = metrics(labels, baseline)
    macro_auc = float(np.mean([f['cnn']['roc_auc'] for f in folds]))
    macro_base = float(np.mean([f['baseline']['roc_auc'] for f in folds]))
    rng = np.random.default_rng(412)
    bootstrap = rng.choice([f['cnn']['roc_auc'] for f in folds], size=(2000, len(folds)), replace=True).mean(1)
    eligible = macro_auc >= .65 and cnn['pr_auc'] >= .60 and macro_auc >= macro_base and cnn['log_loss'] < .6932
    final_config = min(CONFIGS, key=lambda c: np.mean([s['validation_log_loss'] for s in selections if s['config']==c]))
    epochs = max(1, int(np.median([s['epochs'] for s in selections if s['config']==final_config])))
    ensemble_states, final_predictions = [], []
    for seed in [701, 702, 703]:
        model, _, _ = fit(x, y, g, np.arange(len(rows)), None, final_config, seed, epochs)
        ensemble_states.append(model.state_dict())
        final_predictions.append(probability(model, x, g, np.arange(len(rows))))
    checkpoint = ARTIFACTS/'national_fusion.pt'
    torch.save(ensemble_states, checkpoint)
    report = dict(version='national-fusion-region-audit-v2', preprocessing_version=VERSION,
                  eligible_for_deployment=eligible, training_samples=len(rows), occurrences=int(labels.sum()),
                  backgrounds=int((1-labels).sum()), cnn_oof=cnn, spectral_baseline_oof=base,
                  final_fit_training_metrics=metrics(labels, np.mean(final_predictions, axis=0)),
                  seasonal_pair_exclusions=seasonal_exclusions, paired_date_gap_limit_days=60,
                  macro_region_roc_auc=macro_auc, baseline_macro_region_roc_auc=macro_base, folds=folds,
                  macro_region_auc_bootstrap_95_interval=np.quantile(bootstrap, [.025,.975]).tolist(),
                  training_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
                  training_array_sha256=hashlib.sha256((folder/'images.npy').read_bytes()).hexdigest(),
                  final_config=final_config, final_epochs=epochs, ensemble_size=3,
                  model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  evaluation_scope='Nested leave-one-supplied-state-out, 5km exclusion buffer, random initial weights per fold, normalization fitted on training only; all augmentations follow splitting.',
                  label_scope='Provided occurrences versus sampled unknown backgrounds; not verified absence or mineral reserves.',
                  calibration='Uncalibrated screening index; no occurrence probability claim.',
                  limitations=['47 or fewer distinct sites; some states contain only 1-2 occurrences.',
                               'Ladakh imagery failed clear-pixel quality checks; no Ladakh validation.',
                               'Backgrounds are sampled 12km away, primarily in one bearing; sampling design is limited.',
                               '2025 reference imagery; other dates and seasonal shifts not validated.',
                               'Supplied occurrence coordinate provenance unverified.',
                               'Background pixels may contain undiscovered mineralization.'])
    (ARTIFACTS/'national_fusion_report.json').write_text(json.dumps(report, indent=2))
    for row, score, p in zip(rows, oof, baseline):
        row.update(oof_score=float(score), baseline_oof_score=float(p))
    (folder/'predictions.json').write_text(json.dumps(rows, indent=2))
    print('FINAL', json.dumps({k: report[k] for k in ['eligible_for_deployment','cnn_oof','spectral_baseline_oof','macro_region_roc_auc']}), flush=True)


if __name__ == '__main__':
    main()
