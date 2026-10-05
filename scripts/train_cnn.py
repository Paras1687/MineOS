"""Audit, spatial grouping, train, separate calibration, and once-only held-out evaluation."""
import argparse, hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from backend.config import ARTIFACTS, RASTERS
from backend.models.geo_intelligence.raster import read_patch, normalize, site_key, BANDS
from backend.models.geo_intelligence.network import ProspectivityCNN

def report(y,p):
    pred=p>=.5
    return dict(n=len(y),positive=int(sum(y)),prevalence=float(np.mean(y)),pr_auc=float(average_precision_score(y,p)),
        roc_auc=float(roc_auc_score(y,p)),precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        brier=float(brier_score_loss(y,p)),confusion_matrix=confusion_matrix(y,pred).tolist())

def prepare(root,artifact_dir=ARTIFACTS):
    root=Path(root);artifact_dir=Path(artifact_dir);artifact_dir.mkdir(parents=True,exist_ok=True)
    xs=[]; records=[]; errors=[]; seen=set()
    # If a verified-barren folder is present, use it explicitly and exclude
    # Negative_Unknowns; never silently relabel unknown examples as negatives.
    barren=root/'Negative_True_Barren'
    negative_dir=barren if barren.is_dir() else root/'Negative'
    files=sorted((root/'Positive').glob('*.tif'))+sorted(negative_dir.glob('*.tif'))
    for i,p in enumerate(files):
        try:
            x,m=read_patch(p)
            digest=hashlib.sha256(np.nan_to_num(x).tobytes()).hexdigest()
            if digest in seen: raise ValueError('Duplicate pixels')
            seen.add(digest)
            records.append(dict(path=str(p),label=int(p.parent.name.lower()=='positive'),site=site_key(p.stem),
                                date=(re.search(r'\d{4}-\d{2}-\d{2}',p.name) or [''])[0],**m))
            xs.append(normalize(x))
        except Exception as e: errors.append(dict(file=str(p),reason=str(e)))
        if i%200==0: print('audit',i,'/',len(files),flush=True)
    if len(xs)<100: raise RuntimeError('Insufficient valid imagery')
    d=pd.DataFrame(records); parent=list(range(len(d)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b): parent[find(a)]=find(b)
    for _,g in d.groupby('site'):
        for j in g.index[1:]: union(int(g.index[0]),int(j))
    coords=np.radians(d[['latitude','longitude']].values)
    # Nearby sites and background patches cannot cross splits, including overlapping footprints.
    tree=BallTree(coords,metric='haversine')
    for i,neighbors in enumerate(tree.query_radius(coords,r=15000/6371000)):
        for j in neighbors: union(i,int(j))
    d['group']=[find(i) for i in range(len(d))]
    indices=np.arange(len(d))
    def split(idx,fraction,seed):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(idx,groups=d.iloc[idx].group))
        return idx[a],idx[b]
    remainder,test=split(indices,.15,43)
    remainder,cal=split(remainder,.12,44)
    train,val=split(remainder,.18,45)
    d['split']=''
    for name,idx in [('train',train),('validation',val),('calibration',cal),('test',test)]:
        d.loc[idx,'split']=name
        if d.loc[idx,'label'].nunique()!=2: raise RuntimeError(f'{name} lacks both classes')
    d.to_json(artifact_dir/'raster_manifest.json',orient='records',indent=2)
    np.save(artifact_dir/'patches.npy',np.stack(xs))
    label_limit=('User-designated true-barren terrain; barren status is not independently drill/assay verified' if negative_dir.name=='Negative_True_Barren'
                 else 'Negative points are sampled background, not verified barren geology')
    audit=dict(total_files=len(files),accepted=len(d),rejected=errors,groups=int(d.group.nunique()),
               split_counts=d.groupby(['split','label']).size().unstack(fill_value=0).to_dict('index'),
               footprint_m=1280,input_size=64,bands=BANDS,spatial_separation_m=15000,
               cloud_mask='Source SCL/cloud mask was not supplied; residual cloud risk remains',
               negative_folder=str(negative_dir),unknown_negative_folder_excluded=(root/'Negative_Unknowns').is_dir(),
               label_limit=label_limit)
    (artifact_dir/'raster_audit.json').write_text(json.dumps(audit,indent=2))
    return d

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default=str(RASTERS)); p.add_argument('--artifact-dir',default=str(ARTIFACTS)); p.add_argument('--epochs',type=int,default=24); p.add_argument('--prepare-only',action='store_true'); a=p.parse_args()
    out=Path(a.artifact_dir);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    d=prepare(a.data_dir,out) if not (out/'patches.npy').exists() else pd.read_json(out/'raster_manifest.json')
    if a.prepare_only:return
    x=torch.from_numpy(np.load(out/'patches.npy')); y=torch.tensor(d.label.values,dtype=torch.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ProspectivityCNN().to(device)
    tr=np.where(d.split=='train')[0]; va=np.where(d.split=='validation')[0]; ca=np.where(d.split=='calibration')[0]; te=np.where(d.split=='test')[0]
    loader=DataLoader(TensorDataset(x[tr],y[tr]),batch_size=48,shuffle=True)
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()),device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    def logits(idx):
        model.eval()
        with torch.no_grad():return np.concatenate([model(x[z].to(device)).cpu().numpy() for z in np.array_split(idx,max(1,(len(idx)+63)//64))])
    best=-1.; wait=0; history=[]
    print('TRAIN',device,'split sizes',len(tr),len(va),len(ca),len(te),flush=True)
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            if torch.rand(())>.5:xb=xb.flip(-1)
            if torch.rand(())>.5:xb=xb.flip(-2)
            xb=xb.rot90(int(torch.randint(0,4,())),(-2,-1))
            opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        v=average_precision_score(y[va].numpy(),logits(va)); history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val_pr_auc=float(v)))
        print(history[-1],flush=True)
        if v>best+.001:
            best=v; wait=0; torch.save(model.state_dict(),out/'cnn.pt')
        else:wait+=1
        if wait>=6:break
    model.load_state_dict(torch.load(out/'cnn.pt',map_location=device,weights_only=True))
    calibrator=LogisticRegression(C=1.0).fit(logits(ca).reshape(-1,1),y[ca].numpy())
    slope=float(calibrator.coef_[0,0]); intercept=float(calibrator.intercept_[0])
    # Calibration cannot reverse ranking; a negative slope indicates unusable transfer.
    if slope<=0: slope=0.; intercept=float(np.log(y[ca].mean()/(1-y[ca].mean())))
    test_logits=logits(te); probs=1/(1+np.exp(-np.clip(slope*test_logits+intercept,-30,30)))
    metrics=report(y[te].numpy(),probs)
    baseline=LogisticRegression(class_weight='balanced',max_iter=1000).fit(x[tr].mean((2,3)).numpy(),y[tr].numpy())
    metrics['spectral_mean_baseline']=report(y[te].numpy(),baseline.predict_proba(x[te].mean((2,3)).numpy())[:,1])
    metadata=dict(model='MineOS compact 9-band CNN',bands=BANDS,size=64,footprint_m=1280,calibration_slope=slope,
                  calibration_intercept=intercept,history=history,test=metrics,device=device,
                  use='Exploration screening only; calibrated to sampled dataset prevalence, not field occurrence probability',
                  field_validated=False)
    (out/'cnn_metadata.json').write_text(json.dumps(metadata,indent=2))
    d.loc[te,['path','label','split','latitude','longitude','group']].assign(score=probs).to_csv(out/'cnn_test_predictions.csv',index=False)
    print('FINAL',json.dumps(metrics),flush=True)

if __name__=='__main__':main()
