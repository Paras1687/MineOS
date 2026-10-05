from functools import lru_cache
import joblib,numpy as np,pandas as pd
from backend.config import ARTIFACTS
from backend.store import machine_rows,records,distance,reserved_machines,machine_inputs
from backend.catalog import catalog

SENSORS=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
@lru_cache(maxsize=4)
def artifact(name):
    p=ARTIFACTS/name
    if not p.exists():raise RuntimeError(f'Model unavailable: {name}; run training first')
    return joblib.load(p)

@lru_cache(maxsize=1)
def health_probabilities():
    rows=machine_rows()
    frames=[__import__('json').loads(r['sensor_json']) for r in rows]
    probs=artifact('health.joblib').predict_proba(pd.DataFrame(frames)[SENSORS])[:,1]
    return {r['id']:float(p) for r,p in zip(rows,probs)}

def fleet(mine_id=None,near_km=None):
    rows=machine_rows(); frames=[__import__('json').loads(r.pop('sensor_json')) for r in rows]
    overrides=machine_inputs()
    for r in rows:
        v=overrides.get(r['id']);
        if v:r.update({k:v[k] for k in ['mine_id','latitude','longitude','standby']})
    cached=health_probabilities(); probs=[cached[r['id']] for r in rows]
    reservations=reserved_machines()
    planned={a['payload'].get('machine_id'):a['mine_id'] for a in records() if a['state']=='accepted' and a['payload'].get('machine_id')}
    mines={m['id']:m for m in catalog()}
    target=mines.get(mine_id)
    for r,s,p in zip(rows,frames,probs):
        role='support'
        typ=s['Machine Type'].lower()
        if any(x in typ for x in ['load haul','side discharge','excavator']):role='loader'
        elif 'tipper' in typ:role='haulage'
        elif 'crushing' in typ:role='crusher'
        elif 'drill' in typ:role='drill'
        r.update(name=s['Machine Type'],type=s['Machine Type'],role=role,risk_score=float(p),
            status='inspect' if p>=.15 else 'operational',health='high risk' if p>=.5 else ('watch' if p>=.15 else 'lower risk'),
            eligible=bool(p<.15 and r['standby'] and r['id'] not in reservations),reserved=r['id'] in reservations,
            recorded_benchmark_failure=int(s.get('Machine failure',0)),sensors={k:s[k] for k in SENSORS},mine_name=mines.get(r['mine_id'],{}).get('name','Unknown'),
            capacity_tph={'loader':35,'haulage':40,'crusher':100,'drill':0,'support':0}[role],
            planned_destination_id=planned.get(r['id']),planned_destination=mines.get(planned.get(r['id']),{}).get('name'),capacity_source='Editable demo engineering assumptions',source='AI4I synthetic sensors; simulated mine assignment and coordinates',
            observation_time=None,location_live=False,rul_days=None)
        r['mapping_source']=overrides.get(r['id'],{}).get('source','Simulated assignment');r['mapping_date']=overrides.get(r['id'],{}).get('observed_at');r['asset_reference']=overrides.get(r['id'],{}).get('asset_reference')
        if r['id'] in overrides:r['capacity_tph']=overrides[r['id']]['capacity_tph'];r['capacity_source']='Operator input: '+overrides[r['id']]['source']
        if target:r['distance_km']=round(distance(target,r),2)
    if mine_id:
        rows=[r for r in rows if (r['distance_km']<=near_km if near_km is not None else r['mine_id']==mine_id)]
    by_type={}
    for r in rows:
        v=by_type.setdefault(r['type'],dict(total=0,operational=0,inspect=0,standby=0))
        v['total']+=1;v[r['status']]+=1;v['standby']+=int(r['standby'])
    return dict(machines=rows,summary=dict(total=len(rows),operational=sum(r['status']=='operational' for r in rows),
        inspect=sum(r['status']=='inspect' for r in rows),available_standby=sum(r['eligible'] for r in rows),by_type=by_type),
        source='Demo assignments; sensor-only benchmark model. No live GPS or validated time-to-failure.')
