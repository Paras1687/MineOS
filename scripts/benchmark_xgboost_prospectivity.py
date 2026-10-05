"""Compare a compact XGBoost raster-summary baseline to the CNN split.

The supplied split is spatially grouped. Hyperparameters are selected on the
validation split; the held-out test split is scored once after selection.
This script does not replace/deploy the CNN.
"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from xgboost import XGBClassifier
from backend.config import ARTIFACTS

def features(x):
    """Summarize each band's valid-pixel distribution without label leakage."""
    rows=[]
    for patch in x:
        valid=np.isfinite(patch).all(axis=0)&(patch[:6]!=0).any(axis=0)
        vals=patch[:,valid]
        if vals.shape[1]==0:
            vals=np.zeros((patch.shape[0],1),dtype=np.float32)
        q=np.nanquantile(vals,[.1,.25,.5,.75,.9],axis=1).T
        stat=np.concatenate([np.nanmean(vals,axis=1)[:,None],
                             np.nanstd(vals,axis=1)[:,None],q],axis=1).reshape(-1)
        rows.append(np.r_[stat,float(valid.mean())])
    return np.asarray(rows,dtype=np.float32)

def metrics(y,p):
    return dict(n=int(len(y)),positive=int(np.sum(y)),prevalence=float(np.mean(y)),
        roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)))

def group_bootstrap_delta(y,px,pc,groups,seed=42,reps=2000):
    """Paired bootstrap by spatial group, not by individual image patch."""
    rng=np.random.default_rng(seed); unique=np.unique(groups); diffs=[]
    members={g:np.flatnonzero(groups==g) for g in unique}
    for _ in range(reps):
        chosen=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([members[g] for g in chosen])
        if np.unique(y[idx]).size<2:continue
        diffs.append(roc_auc_score(y[idx],px[idx])-roc_auc_score(y[idx],pc[idx]))
    return dict(test_spatial_groups=int(len(unique)),valid_replicates=len(diffs),
        delta_vs_cnn_95pct=[float(v) for v in np.quantile(diffs,[.025,.975])] if diffs else None)

def main():
    d=pd.read_json(ARTIFACTS/'raster_manifest.json')
    x=np.load(ARTIFACTS/'patches.npy',mmap_mode='r')
    X=features(x); y=d.label.to_numpy(dtype=np.int32)
    tr=np.flatnonzero(d.split.to_numpy()=='train')
    va=np.flatnonzero(d.split.to_numpy()=='validation')
    te=np.flatnonzero(d.split.to_numpy()=='test')
    imbalance=float((y[tr]==0).sum()/(y[tr]==1).sum())
    configs=[
        dict(max_depth=2,learning_rate=.03,n_estimators=500,min_child_weight=8,reg_lambda=5),
        dict(max_depth=3,learning_rate=.03,n_estimators=500,min_child_weight=8,reg_lambda=8),
        dict(max_depth=2,learning_rate=.06,n_estimators=300,min_child_weight=12,reg_lambda=10),
        dict(max_depth=3,learning_rate=.06,n_estimators=300,min_child_weight=12,reg_lambda=10),
    ]
    trials=[]; best=None
    for i,cfg in enumerate(configs,1):
        model=XGBClassifier(**cfg,objective='binary:logistic',eval_metric='auc',
            scale_pos_weight=imbalance,subsample=.85,colsample_bytree=.9,
            n_jobs=4,random_state=2026,tree_method='hist',early_stopping_rounds=40)
        model.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],verbose=False)
        vp=model.predict_proba(X[va])[:,1]
        report=dict(config=cfg,best_iteration=int(model.best_iteration),validation=metrics(y[va],vp))
        trials.append(report); print('CANDIDATE',i,json.dumps(report),flush=True)
        if best is None or report['validation']['roc_auc']>best[0]:best=(report['validation']['roc_auc'],model,report)
    _,model,selected=best
    xp=model.predict_proba(X[te])[:,1]
    test=metrics(y[te],xp)
    cnn=json.loads((ARTIFACTS/'cnn_metadata.json').read_text())['test']
    cnn_rows=pd.read_csv(ARTIFACTS/'cnn_test_predictions.csv').set_index('path').score
    cnn_p=np.asarray([cnn_rows[d.iloc[i].path] for i in te],dtype=float)
    test_groups=d.iloc[te].group.to_numpy()
    ci=group_bootstrap_delta(y[te],xp,cnn_p,test_groups)
    india=(d.latitude.between(6,37)&d.longitude.between(68,98)).to_numpy()[te]
    india_rows=int(india.sum())
    india_result=dict(n=india_rows,positive=int(y[te][india].sum()),spatial_groups=int(np.unique(test_groups[india]).size))
    if india_rows and np.unique(y[te][india]).size==2:
        india_result.update(xgboost=metrics(y[te][india],xp[india]),cnn=metrics(y[te][india],cnn_p[india]))
    result=dict(model='XGBoost on nine-band pixel-distribution summaries',features_per_patch=int(X.shape[1]),
        split_source='same 15 km grouped train/validation/calibration/test manifest as CNN',
        selection_metric='validation ROC-AUC; held-out test untouched until final selection',
        validation_candidates=trials,selected=selected,test=test,
        cnn_same_test=dict(roc_auc=cnn['roc_auc'],pr_auc=cnn['pr_auc'],n=cnn['n'],positive=cnn['positive']),
        auc_difference_vs_cnn=float(test['roc_auc']-cnn['roc_auc']),paired_spatial_group_bootstrap=ci,
        india_test_subset=india_result,
        deployment='Benchmark only. Do not deploy unless independent repeat spatial folds and India-local validation support the gain.',
        limitations='Labels are supplied folder classes; background is not verified barren ground. Global test performance does not establish India transfer.')
    path=ARTIFACTS/'xgboost_prospectivity_benchmark.json'
    path.write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
