from functools import lru_cache
from pathlib import Path
import base64,io,json,math,threading
import numpy as np,pandas as pd,torch
from PIL import Image
from rasterio.warp import transform
from backend.config import ARTIFACTS,RASTERS
from backend.catalog import catalog
from backend.store import distance
from backend.operations import artifact
from backend.models.geo_intelligence.raster import read_patch,normalize
from backend.models.geo_intelligence.network import ProspectivityCNN

_lock=threading.Lock()
def footprint(lon,lat,crs,footprint_m=1280):
    x,y=transform('EPSG:4326',crs,[lon],[lat]);x,y=x[0],y[0]
    half=footprint_m/2
    xs=[x-half,x+half,x+half,x-half,x-half];ys=[y-half,y-half,y+half,y+half,y-half]
    longs,lats=transform(crs,'EPSG:4326',xs,ys)
    return dict(type='Polygon',coordinates=[list(map(list,zip(longs,lats)))])

@lru_cache(maxsize=1)
def score_overlay():
    rows={r['path']:r for r in json.loads((ARTIFACTS/'raster_manifest.json').read_text())}
    features=[]
    for p in pd.read_csv(ARTIFACTS/'cnn_test_predictions.csv').to_dict('records'):
        r=rows.get(p['path']);score=float(p['score'])
        if r is None or not math.isfinite(score) or score<=.85:continue
        features.append(dict(type='Feature',geometry=footprint(r['longitude'],r['latitude'],r['crs']),
            properties=dict(name=Path(r['path']).stem,score=score,date=r.get('date'),split=r['split'],
                label=int(r['label']),conflict=int(r['label'])==0,source='Saved held-out CNN evaluation')))
    return dict(type='FeatureCollection',features=features)

@lru_cache(maxsize=1)
def model_bundle():
    if not (ARTIFACTS/'cnn_metadata.json').exists():raise RuntimeError('CNN has not finished training')
    model=ProspectivityCNN();model.load_state_dict(torch.load(ARTIFACTS/'cnn.pt',map_location='cpu',weights_only=True));model.eval()
    return model,json.loads((ARTIFACTS/'cnn_metadata.json').read_text())

def patches():
    path=ARTIFACTS/'raster_manifest.json'
    if not path.exists():return []
    rows=json.loads(path.read_text());out=[]
    for i,r in enumerate(rows):
        out.append(dict(id=str(i),name=Path(r['path']).stem,latitude=r['latitude'],longitude=r['longitude'],
            label=r['label'],split=r['split'],date=r.get('date'),valid_fraction=r['valid_fraction']))
    return out

@lru_cache(maxsize=1)
def _training_sites():
    """Training-only locations used to disclose geographic support at inspection time."""
    path=ARTIFACTS/'raster_manifest.json'
    if not path.exists():return []
    return [r for r in json.loads(path.read_text()) if r.get('split')=='train']

def geographic_support(point,radius_km=25):
    nearby=[]
    for r in _training_sites():
        d=distance(point,{'latitude':r['latitude'],'longitude':r['longitude']})
        if d<=radius_km:nearby.append((r,d))
    positives=[d for r,d in nearby if int(r.get('label',-1))==1]
    backgrounds=[d for r,d in nearby if int(r.get('label',-1))==0]
    return dict(radius_km=radius_km,positive_training_patches=len(positives),
        background_training_patches=len(backgrounds),
        nearest_positive_km=round(min(positives),2) if positives else None,
        nearest_background_km=round(min(backgrounds),2) if backgrounds else None,
        background_label_caveat='Background samples are unlabeled for occurrence absence; they are not verified barren sites.')

def grade_for(site):
    bundle=artifact('grade.joblib'); mapping={'Geology':'geology','Lithology':'lithology','Host_Rock':'host_rock','Formation':'formation','Age':'age'}
    row={col:site.get(mapping[col]) or 'Unknown' for col in bundle['features']}
    unknown=[col for col,v in row.items() if v not in bundle['vocabulary'][col]]
    if len(unknown)>2:return dict(status='unavailable',reason='Geology outside training support',unknown_features=unknown)
    value=float(bundle['model'].predict(pd.DataFrame([row]))[0]);error=bundle['error']
    return dict(status='experimental analogue estimate',grade_pct=round(value,2),range_pct=[round(max(0,value-error),2),round(min(65,value+error),2)],
        unknown_features=unknown,warning='Range is screening error from supplied source labels, not a field-calibrated interval. Surface geology cannot establish depth or mineable reserve.')

def evaluate(path,lon=None,lat=None,extra=None):
    arr,meta=read_patch(path,lon,lat);model,metadata=model_bundle()
    x=torch.from_numpy(normalize(arr)[None]);x.requires_grad_(True)
    with _lock:
        model.zero_grad(set_to_none=True)
        feat=model.features(x);feat.retain_grad();logit=model.head(model.pool(feat).flatten(1)).squeeze()
        logit.backward(); weights=feat.grad.mean((2,3),keepdim=True)
        cam=torch.relu((weights*feat).sum(1))[0].detach().numpy()
        if cam.max()>0:cam/=cam.max()
        # Match the held-out evaluation path: reported test metrics and CSV scores
        # apply this separately fitted calibration layer to the CNN logit.
        calibrated_logit=(metadata.get('calibration_slope',1.0)*float(logit.detach())+
                          metadata.get('calibration_intercept',0.0))
        raw_score=float(1/(1+math.exp(-np.clip(calibrated_logit,-30,30))))
    rgb=arr[[2,1,0]].transpose(1,2,0);rgb=np.nan_to_num(rgb)
    rgb=np.clip(rgb/np.maximum(np.percentile(rgb,98,axis=(0,1)),1),0,1)
    heat=np.asarray(Image.fromarray((cam*255).astype('uint8')).resize((64,64)))/255
    overlay=.65*rgb+.35*np.stack([heat,np.zeros_like(heat),1-heat],axis=-1)
    stream=io.BytesIO();Image.fromarray((np.clip(overlay,0,1)*255).astype('uint8')).resize((320,320)).save(stream,format='PNG')
    point=dict(longitude=meta['longitude'],latitude=meta['latitude']);near=[dict(m) for m in sorted([m for m in catalog() if m['target']==1],key=lambda s:distance(s,point))[:3]]
    for s in near:s['distance_km']=round(distance(s,point),2)
    known=near[0] if near and near[0]['distance_km']<=.1 else None
    metrics=metadata['test'];policy=metadata.get('deployment_policy',{})
    support=geographic_support(point)
    sample_label=(extra or {}).get('sample_label')
    label_conflict=(sample_label is not None and ((int(sample_label)==0 and raw_score>=.5) or
                    (int(sample_label)==1 and raw_score<.5)))
    label_note=None
    if sample_label is not None:
        split=(extra or {}).get('split')
        if split!='test':
            label_note='IN-SAMPLE ONLY: this TIFF was used for training, calibration or model selection. Its score and agreement with the supplied label are not independent evidence.'
            if label_conflict and int(sample_label)==0:label_note+=' The CNN also contradicts its supplied barren label (false positive).'
            elif label_conflict:label_note+=' The CNN also contradicts its supplied positive label.'
        elif label_conflict and int(sample_label)==0:
            label_note='Model conflict: this supplied TIFF is labelled barren, but the CNN is above its 0.5 screening threshold. This is a false positive on a labelled sample, not high-confidence manganese evidence.'
        elif label_conflict:
            label_note='Model conflict: this supplied TIFF is positive-labelled, but the CNN is below its 0.5 screening threshold.'
        else:
            label_note='CNN threshold agrees with the supplied TIFF label; this is not independent field verification.'
    return dict(prospectivity_score=round(raw_score,4),score_status='calibrated-to-sample experimental index; not a probability',
        score_label='Calibrated sample score index (0–100)',
        score_meaning=policy.get('warning','Experimental index only; not a field-validated occurrence probability.'),
        geographic_support=support,field_validated=False,
        location=point,raster=meta,footprint=footprint(meta['longitude'],meta['latitude'],meta['crs']),source=extra or {'kind':'local supplied GeoTIFF'},heatmap='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode(),
        heatmap_note='Grad-CAM shows model-sensitive areas, not manganese boundaries or geological proof',
        nearby_occurrences=near,grade=grade_for(known) if known else dict(status='unavailable',reason='No verified local geology supplied for this point'),
        resource=dict(status='unavailable',reason='Point imagery cannot establish ore volume, tonnage or economically recoverable reserve'),
        test_metrics=metadata['test'],deployment_policy=policy,
        sample_label=int(sample_label) if sample_label is not None else None,
        sample_label_name=(extra or {}).get('sample_name'),sample_label_note=label_note,
        sample_split=(extra or {}).get('split'),
        sample_independent=((extra or {}).get('split')=='test'),
        sample_label_conflict=bool(label_conflict))

def _local_heldout_point(lon,lat,fallback_warning=None):
    candidates=sorted(patches(),key=lambda r:distance(r,dict(longitude=lon,latitude=lat)))
    raw=json.loads((ARTIFACTS/'raster_manifest.json').read_text()) if candidates else []
    for independent in (True,False):
        for r in candidates:
            gap=distance(r,dict(longitude=lon,latitude=lat))
            if gap>3:break
            if (r.get('split')=='test')!=independent:continue
            path=Path(raw[int(r['id'])]['path'])
            if not path.exists():
                # Preserve class subdirectory when the supplied raster folder is relocated.
                path=RASTERS/path.parent.name/path.name
            warning=fallback_warning
            if not independent:
                disclosure='No held-out image was nearby. This supplied patch is in the '+str(r['split'])+' split; the CNN used it for training, calibration or model selection, so this score is not independent evidence.'
                warning=(warning+' ' if warning else '')+disclosure
            try:return evaluate(path,lon,lat,{'kind':'local raster','date':r['date'],'split':r['split'],
                'sample_label':int(r['label']),'sample_name':r['name'],
                'distance_km':round(gap,2),'fallback_warning':warning})
            except (ValueError,OSError):continue
    raise ValueError('No supplied TIFF within 3 km. Choose a labelled map patch or use cloud imagery when internet is available; no score can be supported for this location.')

def predict_point(lon,lat,cloud=False):
    if not cloud:return _local_heldout_point(lon,lat)
    from backend.cloud import fetch_patch
    try:
        path,meta=fetch_patch(lon,lat);return evaluate(path,lon,lat,meta)
    except Exception as exc:
        message=str(exc)
        blocked=('10013' in message or 'forbidden by its access permissions' in message.lower())
        warning=('Cloud satellite access is blocked on this computer/session. Showing nearby local held-out imagery instead; its score applies to that supplied patch, not exactly to the clicked coordinate.'
                 if blocked else 'Cloud imagery was unavailable. Showing nearby local held-out imagery instead; its score applies to that supplied patch, not exactly to the clicked coordinate.')
        try:return _local_heldout_point(lon,lat,warning)
        except ValueError as local_exc:
            if blocked:
                raise RuntimeError('Windows blocked the cloud imagery connection, and there is no held-out local image within 3 km. Uncheck “Fetch cloud imagery” and inspect a supplied held-out map patch, or retry when cloud access is available. No point score was generated.') from local_exc
            raise RuntimeError(f'Cloud imagery failed: {message[:220]}. {local_exc}') from exc

def ranking():
    metadata=model_bundle()[1];metrics=metadata['test']
    if not metadata.get('deployment_policy',{}).get('comparative_ranking_enabled',False):return []
    if metrics['roc_auc']<.60 or metrics['pr_auc']<metrics['prevalence']*1.20:return []
    path=ARTIFACTS/'prospect_ranking.json'
    return json.loads(path.read_text()) if path.exists() else []


def scan_nearby(lon,lat):
    import rasterio
    from backend.cloud import fetch_patch
    path,source=fetch_patch(lon,lat,size=576,min_valid=.3)
    model,metadata=model_bundle();features=[];missing=[];arrays=[];positions=[]
    with rasterio.open(path) as src:
        image=src.read();crs=str(src.crs)
        for dy in range(-3,4):
            for dx in range(-3,4):
                if math.hypot(dx*1280,dy*1280)>5000:continue
                row,col=288+dy*64,288+dx*64
                x,y=src.transform*(col,row)
                longs,lats=transform(crs,'EPSG:4326',[x],[y]);plon,plat=longs[0],lats[0]
                actual=distance(dict(longitude=lon,latitude=lat),dict(longitude=plon,latitude=plat))
                if actual>5:continue
                patch=image[:,row-32:row+32,col-32:col+32]
                valid=np.all(np.isfinite(patch),axis=0)&np.any(patch[:6]!=0,axis=0)
                pos=dict(longitude=plon,latitude=plat,distance_km=round(actual,3),valid_fraction=float(valid.mean()))
                if valid.mean()<.8:missing.append(pos);continue
                arrays.append(normalize(patch));positions.append(pos)
    if arrays:
        with _lock,torch.inference_mode():
            logits=model(torch.from_numpy(np.stack(arrays))).reshape(-1)
            scores=torch.sigmoid(logits*metadata.get('calibration_slope',1)+metadata.get('calibration_intercept',0)).tolist()
        for pos,score in zip(positions,scores):
            features.append(dict(type='Feature',geometry=footprint(pos['longitude'],pos['latitude'],crs),properties={**pos,'score':round(score,4),'name':'Nearby grid patch','date':source['acquisition_date'],'source':source['source'],'split':'cloud inspection','high_score':score>=.9}))
    hits=sum(f['properties']['high_score'] for f in features)
    message=(f'{hits} sampled patches scored 90+ within 5 km.' if hits else 'No sampled usable patches scored 90+ within 5 km.')
    if missing:message+=' Scan incomplete: some grid patches had insufficient clear imagery.'
    message+=' This is 1.28 km grid screening, not an exhaustive mineral survey; it cannot confirm reserves, absence of manganese or extraction feasibility.'
    return dict(type='FeatureCollection',features=features,missing=missing,source=source,radius_km=5,grid_spacing_m=1280,total=len(features)+len(missing),scored=len(features),high_score_count=hits,incomplete=bool(missing),message=message)
