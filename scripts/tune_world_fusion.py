import sys,json,zipfile,hashlib,copy,time,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,log_loss,brier_score_loss
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.config import ARTIFACTS
from backend.models.geo_intelligence.world_fusion import WorldFusionCNN

def scores(model,x,g,idx):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x[j],g[j]) for j in np.array_split(idx,max(1,int(np.ceil(len(idx)/64))))]).numpy()

def probability(logits,temperature=1):
    return 1/(1+np.exp(-np.clip(logits/temperature,-30,30)))

def metrics(y,p,threshold=.5):
    return dict(n=len(y),positives=int(y.sum()),roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
        accuracy=float(accuracy_score(y,p>=threshold)),balanced_accuracy=float(balanced_accuracy_score(y,p>=threshold)),
        precision=float(precision_score(y,p>=threshold,zero_division=0)),recall=float(recall_score(y,p>=threshold)),
        f1=float(f1_score(y,p>=threshold)),log_loss=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)),
        confusion_matrix=confusion_matrix(y,p>=threshold).tolist(),threshold=float(threshold))

def main():
    torch.set_num_threads(4)
    bundle=Path(sys.argv[1])
    with zipfile.ZipFile(bundle) as z:
        a=np.load(z.open('world_unknown_training/arrays.npz'))
        x=torch.from_numpy(a['x'].transpose(0,3,1,2).copy());g=torch.from_numpy(a['g'].reshape(-1,1).copy())
        y=a['y'].astype('int64');rows=json.loads(z.read('world_unknown_training/manifest.json'))
    coords=np.radians([[r['latitude'],r['longitude']] for r in rows]);tree=BallTree(coords,metric='haversine');parent=list(range(len(y)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,neighbours in enumerate(tree.query_radius(coords,r=100/6371.0088)):
        for j in neighbours:parent[root(int(j))]=root(i)
    groups=np.array([root(i) for i in range(len(y))])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
    te,va=folds[0][1],folds[1][1];tr=np.setdiff1d(np.arange(len(y)),np.r_[te,va])
    for aa,bb in [(tr,va),(tr,te),(va,te)]:
        if set(groups[aa])&set(groups[bb]):raise ValueError('Spatial split leakage')
        if len(np.unique(y[aa]))!=2 or len(np.unique(y[bb]))!=2:raise ValueError('Missing class in spatial split')
    minimum=float(BallTree(coords[tr],metric='haversine').query(coords[te])[0].min()*6371.0088)
    mean,var=float(g[tr].mean()),float(g[tr].var(unbiased=False));yt=torch.from_numpy(y.astype('float32'))
    configs=[dict(name='baseline',epochs=24,lr=.0005,decay=0.,augment=False,normalize_first=False),
             dict(name='regularized',epochs=36,lr=.0005,decay=.001,augment=True,normalize_first=True),
             dict(name='low_rate',epochs=40,lr=.00025,decay=.005,augment=True,normalize_first=True)]
    candidates=[];states={}
    for config in configs:
        torch.manual_seed(42);rng=np.random.default_rng(42)
        model=WorldFusionCNN(mean,var,config['normalize_first'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['decay'])
        weights=torch.tensor([len(tr)/(2*(y[tr]==c).sum()) for c in (0,1)],dtype=torch.float32)
        best=float('inf');stale=0;history=[]
        for epoch in range(config['epochs']):
            model.train();total=0.
            for idx in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/32))):
                bx=x[idx]
                if config['augment']:
                    bx=torch.rot90(bx,int(rng.integers(4)),(2,3))
                    if rng.random()<.5:bx=bx.flip(3)
                optimizer.zero_grad(set_to_none=True)
                logits=model(bx,g[idx]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,yt[idx],reduction='none')*weights[yt[idx].long()]).mean()
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();total+=float(loss.detach())*len(idx)
            vp=probability(scores(model,x,g,va));vl=float(log_loss(y[va],vp));history.append(dict(epoch=epoch+1,training_loss=total/len(tr),validation_log_loss=vl,validation_auc=float(roc_auc_score(y[va],vp))))
            if vl<best-1e-4:best=vl;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
            else:stale+=1
            if (epoch+1)%5==0:print(config['name'],epoch+1,'validation',round(vl,4),flush=True)
            if stale>=7:break
        model.load_state_dict(state);states[config['name']]=state
        candidates.append(dict(config=config,best_epoch=best_epoch,validation=metrics(y[va],probability(scores(model,x,g,va))),history=history))
        print('DONE',config['name'],best_epoch,round(best,5),flush=True)
    chosen=min(candidates,key=lambda c:c['validation']['log_loss']);name=chosen['config']['name']
    model=WorldFusionCNN(mean,var,chosen['config']['normalize_first']);model.load_state_dict(states[name])
    vl=scores(model,x,g,va)
    temperature=float(minimize_scalar(lambda t:log_loss(y[va],probability(vl,t)),bounds=(.5,5),method='bounded').x)
    vp=probability(vl,temperature)
    threshold=float(max(np.linspace(.1,.9,81),key=lambda t:balanced_accuracy_score(y[va],vp>=t)))
    tp=probability(scores(model,x,g,te),temperature)
    baseline=WorldFusionCNN(mean,var);baseline.load_state_dict(states['baseline'])
    gp=g.clone();gp[te]=mean
    report=dict(selected=name,selection_basis='Lowest validation log loss; test evaluated only after selection',
        splits={k:dict(n=len(idx),positive=int(y[idx].sum()),spatial_groups=len(set(groups[idx])),indices=idx.tolist()) for k,idx in [('train',tr),('validation',va),('test',te)]},
        minimum_train_test_km=minimum,candidates=candidates,temperature=temperature,threshold=threshold,
        train=metrics(y[tr],probability(scores(model,x,g,tr),temperature),threshold),validation=metrics(y[va],vp,threshold),
        test=metrics(y[te],tp,threshold),baseline_test=metrics(y[te],probability(scores(baseline,x,g,te))),
        gravity_mean_ablation_test_auc=float(roc_auc_score(y[te],probability(scores(model,x,gp,te),temperature))),
        limitations=['Retrospective spatial holdout on previously supplied data, not a new external field test.',
                    'Unknown-background labels are not verified absence. Legacy raster scale/alignment and coordinate provenance remain unverified.',
                    'No guarantee of India field performance or absence of overfitting outside this sample.'])
    # Refit the selected recipe on every supplied sample for deployment. Keep
    # holdout metrics above as evaluation evidence from before this refit.
    deploy_mean,deploy_var=float(g.mean()),float(g.var(unbiased=False))
    deployment=WorldFusionCNN(deploy_mean,deploy_var,chosen['config']['normalize_first'])
    optimizer=torch.optim.AdamW(deployment.parameters(),lr=chosen['config']['lr'],weight_decay=chosen['config']['decay'])
    weights=torch.tensor([len(y)/(2*(y==c).sum()) for c in (0,1)],dtype=torch.float32)
    rng=np.random.default_rng(42042)
    for epoch in range(chosen['best_epoch']):
        deployment.train()
        for idx in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/32))):
            bx=x[idx]
            if chosen['config']['augment']:
                bx=torch.rot90(bx,int(rng.integers(4)),(2,3))
                if rng.random()<.5:bx=bx.flip(3)
            optimizer.zero_grad(set_to_none=True)
            logits=deployment(bx,g[idx]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,yt[idx],reduction='none')*weights[yt[idx].long()]).mean()
            if not torch.isfinite(loss):raise ValueError('Non-finite full-data refit loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(deployment.parameters(),2.);optimizer.step()
        print('FULL REFIT',epoch+1,'/',chosen['best_epoch'],flush=True)
    deployment.eval()
    report['deployment_refit']=dict(samples=len(y),positives=int(y.sum()),epochs=chosen['best_epoch'],
        validation_claim='The refit uses all samples; the spatial holdout metrics above belong to model selection before refit.')
    backup=ROOT/'backend/experiments/before_final_tuning';backup.mkdir(parents=True,exist_ok=True)
    for f in ['world_fusion.pt','world_fusion_metadata.json']:
        if not (backup/f).exists():shutil.copy2(ARTIFACTS/f,backup/f)
    torch.save(deployment.state_dict(),ARTIFACTS/'world_fusion.pt')
    metadata=dict(version='moil-world-fusion-spatial-full-v3',architecture='Sentinel-2 CNN + normalized Bouguer fusion',
        architecture_options={'gravity_dropout_after_norm':chosen['config']['normalize_first']},gravity_mean=deploy_mean,gravity_variance=deploy_var,
        training_samples=len(y),positive_samples=int(y.sum()),unknown_background_samples=int((y==0).sum()),
        full_dataset_fit=True,epochs=chosen['best_epoch'],calibrated=False,calibration_scope='Uncalibrated full-data refit; score is not a field probability',
        temperature=1.0,decision_threshold=.5,test=report['test'],validation=report['validation'],
        evaluation_scope='Metrics are from spatial holdout model selection before final all-data refit; not an independent evaluation of the deployed full-data checkpoint.',
        model_sha256=hashlib.sha256((ARTIFACTS/'world_fusion.pt').read_bytes()).hexdigest(),
        gravity_sha256=hashlib.sha256((ARTIFACTS/'world_bouguer.grd').read_bytes()).hexdigest(),
        warning='; '.join(report['limitations']))
    (ARTIFACTS/'world_fusion_metadata.json').write_text(json.dumps(metadata,indent=2))
    (ARTIFACTS/'world_fusion_validation.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps({k:report[k] for k in ['selected','test','baseline_test','minimum_train_test_km']}),flush=True)

if __name__=='__main__':main()
