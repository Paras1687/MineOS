"""Random Forest baseline on the CNN's existing spatially grouped image split."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from backend.config import ARTIFACTS
from benchmark_xgboost_prospectivity import features,metrics,group_bootstrap_delta

def main():
    d=pd.read_json(ARTIFACTS/'raster_manifest.json')
    x=np.load(ARTIFACTS/'patches.npy',mmap_mode='r'); X=features(x)
    y=d.label.to_numpy(dtype=np.int32)
    tr=np.flatnonzero(d.split.to_numpy()=='train')
    va=np.flatnonzero(d.split.to_numpy()=='validation')
    te=np.flatnonzero(d.split.to_numpy()=='test')
    configs=[
        dict(n_estimators=500,max_depth=8,min_samples_leaf=3,max_features='sqrt'),
        dict(n_estimators=500,max_depth=12,min_samples_leaf=5,max_features=.7),
        dict(n_estimators=500,max_depth=None,min_samples_leaf=5,max_features=.7),
        dict(n_estimators=500,max_depth=6,min_samples_leaf=10,max_features=.5),
    ]
    trials=[];best=None
    for i,cfg in enumerate(configs,1):
        model=RandomForestClassifier(**cfg,class_weight='balanced_subsample',
            n_jobs=4,random_state=2026)
        model.fit(X[tr],y[tr]);p=model.predict_proba(X[va])[:,1]
        report=dict(config=cfg,validation=metrics(y[va],p));trials.append(report)
        print('CANDIDATE',i,json.dumps(report),flush=True)
        if best is None or report['validation']['roc_auc']>best[0]:best=(report['validation']['roc_auc'],model,report)
    _,model,selected=best
    rp=model.predict_proba(X[te])[:,1]; test=metrics(y[te],rp)
    cnn=json.loads((ARTIFACTS/'cnn_metadata.json').read_text())['test']
    xgb=json.loads((ARTIFACTS/'xgboost_prospectivity_benchmark.json').read_text())
    cnn_rows=pd.read_csv(ARTIFACTS/'cnn_test_predictions.csv').set_index('path').score
    cp=np.asarray([cnn_rows[d.iloc[i].path] for i in te],dtype=float)
    groups=d.iloc[te].group.to_numpy()
    ci=group_bootstrap_delta(y[te],rp,cp,groups)
    india=(d.latitude.between(6,37)&d.longitude.between(68,98)).to_numpy()[te]
    india_result=dict(n=int(india.sum()),positive=int(y[te][india].sum()),
        spatial_groups=int(np.unique(groups[india]).size))
    result=dict(model='Random Forest on nine-band pixel-distribution summaries',
        features_per_patch=int(X.shape[1]),
        split_source='same spatially grouped train/validation/calibration/test manifest as CNN and XGBoost',
        selection_metric='validation ROC-AUC; test scored after selection',
        validation_candidates=trials,selected=selected,test=test,
        cnn_same_test=dict(roc_auc=cnn['roc_auc'],pr_auc=cnn['pr_auc'],n=cnn['n']),
        xgboost_same_test=dict(roc_auc=xgb['test']['roc_auc'],pr_auc=xgb['test']['pr_auc'],n=xgb['test']['n']),
        auc_difference_vs_cnn=float(test['roc_auc']-cnn['roc_auc']),
        paired_spatial_group_bootstrap_vs_cnn=ci,india_test_subset=india_result,
        deployment='Benchmark only; India test subset has no patches and background labels are not verified barren ground.')
    path=ARTIFACTS/'random_forest_prospectivity_benchmark.json'
    path.write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
