"""Local advisory API: explicit data provenance and persistent operator decisions."""
import json,uuid,threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from time import monotonic
from typing import Literal
import torch
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import RedirectResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from backend.config import ROOT,ARTIFACTS
from backend.catalog import catalog
from backend.store import initialize,update_action,records,connection,now
from backend.operations import fleet
from backend.planning import predict,alternatives,get_mine
from backend.exploration import patches,predict_point,ranking,grade_for,score_overlay,scan_nearby
from backend.models.geo_intelligence.world_fusion import inspect_point as inspect_world_point, scan_nearby as scan_world_nearby
from backend.weather import forecast as weather_forecast

torch.set_num_threads(1)
pool=ThreadPoolExecutor(max_workers=1);jobs={};job_lock=threading.Lock()
@asynccontextmanager
async def lifespan(app):
    initialize();yield
app=FastAPI(title='MineOS AI',version='2.0.0',lifespan=lifespan)

@app.exception_handler(ValueError)
async def value_error(request,exc):return JSONResponse(status_code=422,content={'detail':str(exc)})
@app.exception_handler(KeyError)
async def key_error(request,exc):return JSONResponse(status_code=404,content={'detail':str(exc)})

class Options(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    period:Literal['shift','daily','weekly','monthly']='daily'
    shift:Literal['Shift-1','Shift-2','Shift-3']='Shift-1'
    start_date:date|None=None
    live_weather:bool=False
    simulate_events:bool=False
    fleet_available_pct_override:float|None=Field(None,ge=0,le=100)
    fleet_risk_pct_override:float|None=Field(None,ge=0,le=100)
    equipment_downtime_hours:float=Field(0,ge=0,le=8)
    mode:Literal['forecast','replay']='forecast'
    replay_block:Literal['Block-A','Block-B','Block-C']='Block-A'
    replay_date:date|None=None
    rainfall_mm:float|None=Field(None,ge=0,le=1000)
    blast_delay_hours:float|None=Field(None,ge=0,le=8)
    target_per_shift_mt:float|None=Field(None,ge=0,le=100000)
    price_per_tonne:float=Field(10000,ge=0,le=1000000)
    variable_cost_per_tonne:float=Field(4000,ge=0,le=1000000)
    stockpile_mt:float=Field(200,ge=0,le=100000)
    stockpile_handling_per_tonne:float=Field(250,ge=0,le=100000)
    processing_headroom_tph:float=Field(20,ge=0,le=10000)
    max_transfer_km:float=Field(200,ge=0,le=2000)
    transport_speed_kph:float=Field(25,gt=0,le=100)
    mobilization_inr:float=Field(8000,ge=0,le=10000000)
    transport_inr_per_km:float=Field(120,ge=0,le=10000)
    extra_shift_hours:float=Field(1,ge=0,le=2)
    overtime_cost_per_hour:float=Field(3000,ge=0,le=1000000)

class Point(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    longitude:float=Field(ge=-180,le=180)
    latitude:float=Field(ge=-90,le=90)
    cloud:bool=False
class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    state:Literal['acknowledged','accepted','rejected','completed']
    reason:str=Field(min_length=3,max_length=1000)
    actor:str=Field(min_length=2,max_length=100)
    actual_recovered_mt:float|None=Field(None,ge=0,le=10000000)
    actual_cost_inr:float|None=Field(None,ge=0,le=1000000000)
class Assay(BaseModel):
    site_id:str
    assay_pct:float=Field(ge=0,le=100)
    notes:str=Field(min_length=5,max_length=1000)

@app.get('/api/v1/health')
def health():
    return dict(status='ready',version='2.0.0',models={s:(ARTIFACTS/f).exists() for s,f in [('cnn','cnn.pt'),('world_fusion','world_fusion.pt'),('health','health.joblib'),('production','production.joblib'),('grade','grade.joblib')]})
@app.get('/api/v1/mines')
def mines():return dict(data=[r for r in catalog() if r['target']==1],source='User master CSV; operational status unverified')
@app.get('/api/v1/exploration/patches')
def patch_index():
    audit=ARTIFACTS/'raster_audit.json'
    return dict(data=patches(),audit=json.loads(audit.read_text()) if audit.exists() else {})
@app.get('/api/v1/exploration/ranking')
def priorities():
    metadata=json.loads((ARTIFACTS/'cnn_metadata.json').read_text());metrics=metadata['test'];policy=metadata.get('deployment_policy',{})
    usable=policy.get('comparative_ranking_enabled',False) and metrics['roc_auc']>=.60 and metrics['pr_auc']>=metrics['prevalence']*1.20
    return dict(data=ranking(),usable=usable,metrics=metrics,
        policy=policy,
        status='Research screening only' if usable else 'Comparative ranking withheld: India field-validation gate is not met',
        warning=policy.get('warning','Rank does not authorise drilling or mining. Train sites are not new discoveries.'))

@app.get('/api/v1/exploration/heatmap')
def exploration_heatmap():return score_overlay()

from backend.estimation import EstimateRequest, estimate, versions

@app.get('/api/v1/exploration/estimation-version')
def estimation_version():
    try:return versions()
    except FileNotFoundError:raise HTTPException(503,'Grade preparation artifacts missing. Run scripts/prepare_grade_reserve.py offline.')

@app.post('/api/v1/exploration/estimate')
def estimate_grade_reserve(request:EstimateRequest):
    try:return estimate(request)
    except FileNotFoundError:raise HTTPException(503,'Grade preparation artifacts missing. Run scripts/prepare_grade_reserve.py offline.')

def run_job(uid,point,nearby=False):
    with job_lock:jobs[uid]['status']='running'
    try:
        result=scan_world_nearby(point.longitude,point.latitude) if nearby else inspect_world_point(point.longitude,point.latitude)
        with job_lock:jobs[uid].update(status='complete',result=result)
    except Exception as exc:
        with job_lock:jobs[uid].update(status='failed',error=str(exc).split('?')[0][:600])
def submit_job(point, nearby=False):
    key=(point.longitude,point.latitude,nearby)
    with job_lock:
        for uid,existing in jobs.items():
            if existing.get('key')==key and (existing['status'] in ['queued','running'] or
                    (existing['status']=='complete' and monotonic()-existing.get('created',0)<600)):
                return dict(job_id=uid)
        if sum(j['status'] in ['queued','running'] for j in jobs.values())>=4:
            raise HTTPException(429,'Four predictions already in progress; wait for an active request to finish')
        for uid in list(jobs):
            if jobs[uid]['status'] in ['complete','failed'] and (len(jobs)>=100 or monotonic()-jobs[uid].get('created',0)>600):
                jobs.pop(uid)
        uid=uuid.uuid4().hex
        jobs[uid]={'id':uid,'status':'queued','key':key,'created':monotonic()}
    pool.submit(run_job,uid,point,nearby)
    return dict(job_id=uid)

@app.post('/api/v1/exploration/predict',status_code=202)
def point_prediction(point:Point):
    return submit_job(point)

@app.post('/api/v1/exploration/nearby',status_code=202)
def nearby_prediction(point:Point):
    return submit_job(point,True)

@app.get('/api/v1/jobs/{uid}')
def job(uid:str):
    with job_lock:
        if uid not in jobs:raise HTTPException(404,'Prediction job not found')
        return {k:v for k,v in jobs[uid].items() if k not in ('key','created')}
@app.get('/api/v1/mines/{uid}/geology')
def geology(uid:str):
    mine=get_mine(uid);return dict(data=mine,grade_estimate=grade_for(mine),resource_prediction=None,resource_reason='No verified tonnage units or ore volume labels')
@app.get('/api/v1/operational/equipment')
def equipment(mine_id:str|None=None,near_km:float|None=Query(None,ge=0,le=1000)):
    if mine_id:get_mine(mine_id)
    return fleet(mine_id,near_km)
@app.get('/api/v1/weather/{uid}')
def weather(uid:str):
    m=get_mine(uid);return weather_forecast(m['latitude'],m['longitude'])
@app.post('/api/v1/production/{uid}')
def production(uid:str,opts:Options):return predict(uid,resolve_options(uid,opts))
@app.post('/api/v1/decisions/{uid}')
def decisions(uid:str,opts:Options):return alternatives(uid,resolve_options(uid,opts))
@app.post('/api/v1/actions/{uid}')
def action(uid:str,payload:Decision):return update_action(uid,payload.state,payload.reason,payload.actor,payload.actual_recovered_mt,payload.actual_cost_inr)
@app.get('/api/v1/records')
def feedback():
    items=records();completed=[r for r in items if r['state']=='completed']
    return dict(data=items,summary=dict(completed=len(completed),actual_recovered_mt=sum(r['actual_recovered_mt'] for r in completed),actual_cost_inr=sum(r['actual_cost'] for r in completed)))
@app.post('/api/v1/field-results',status_code=201)
def field_result(result:Assay):
    get_mine(result.site_id)
    with connection() as c:cur=c.execute('INSERT INTO field_results(at,site_id,assay_pct,notes) VALUES (?,?,?,?)',(now(),result.site_id,result.assay_pct,result.notes))
    return dict(id=cur.lastrowid,status='queued for validation',model_retrained=False)
@app.get('/api/v1/evidence')
def evidence():
    result={}
    for key,name in [('cnn','cnn_metadata.json'),('dataset','raster_audit.json'),('tabular','tabular_metrics.json')]:
        path=ARTIFACTS/name;result[key]=json.loads(path.read_text()) if path.exists() else {'status':'not trained'}
    world=ARTIFACTS/'world_fusion_validation.json'
    result['multimodal_prospectivity']=json.loads(world.read_text()) if world.exists() else {'status':'not trained'}
    from backend.models.geo_intelligence.world_fusion import model_bundle as active_fusion
    result['active_multimodal_model']=active_fusion()[1]
    national=ARTIFACTS/'national_fusion_report.json'
    result['national_multimodal_validation']=json.loads(national.read_text()) if national.exists() else {'status':'not evaluated'}
    return result
@app.get('/api/v1/overview')
def overview():
    f=fleet();m=[r for r in catalog() if r['target']==1]
    return dict(mines=len(m),india_sites=sum(r['country']=='India' for r in m),fleet=f['summary'],patches=len(patches()),source='Occurrence registry + benchmark sensors + simulated operations',
        alerts=[{'severity':'watch','text':'Exploration CNN is experimental; global test results do not establish India performance.'},
                {'severity':'watch','text':'Machine locations and mine-production mappings are demo assignments.'},
                {'severity':'watch','text':'Verified resource units and current mine production feeds are unavailable.'}])
@app.get('/')
def home():return RedirectResponse('/ui/dashboard.html')
app.mount('/ui',StaticFiles(directory=ROOT/'frontend',html=True),name='frontend')


class EvidenceBase(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    source:str=Field(min_length=5,max_length=500)
    observed_at:date
    operator:str=Field(min_length=2,max_length=100)
class SiteInputs(EvidenceBase):
    mine_type:Literal['Underground','Opencast','Mixed','Unknown']='Unknown'
    depth_m:float|None=Field(None,ge=0,le=5000)
    block:Literal['Block-A','Block-B','Block-C']|None=None
    price_per_tonne:float|None=Field(None,ge=0,le=1000000)
    variable_cost_per_tonne:float|None=Field(None,ge=0,le=1000000)
    stockpile_mt:float|None=Field(None,ge=0,le=100000)
    processing_headroom_tph:float|None=Field(None,ge=0,le=10000)
    stockpile_handling_per_tonne:float|None=Field(None,ge=0,le=100000)
    transport_speed_kph:float|None=Field(None,gt=0,le=100)
    mobilization_inr:float|None=Field(None,ge=0,le=10000000)
    transport_inr_per_km:float|None=Field(None,ge=0,le=10000)
    overtime_cost_per_hour:float|None=Field(None,ge=0,le=1000000)
class MachineInputs(EvidenceBase):
    mine_id:str
    asset_reference:str=Field(min_length=2,max_length=100)
    latitude:float=Field(ge=-90,le=90)
    longitude:float=Field(ge=-180,le=180)
    standby:bool=False
    capacity_tph:float=Field(ge=0,le=10000)
class ShiftLog(EvidenceBase):
    day:date
    shift:Literal['Shift-1','Shift-2','Shift-3']
    actual_mt:float=Field(ge=0,le=1000000)
    target_mt:float=Field(ge=0,le=1000000)
    blast_delay_hours:float=Field(ge=0,le=8)

@app.get('/api/v1/site-inputs/{uid}')
def read_site_inputs(uid:str):
    from backend.store import site_inputs,shift_logs
    get_mine(uid);return dict(inputs=site_inputs(uid),shift_logs=shift_logs(uid))
@app.post('/api/v1/site-inputs/{uid}')
def write_site_inputs(uid:str,data:SiteInputs):
    get_mine(uid)
    with connection() as c:c.execute('INSERT OR REPLACE INTO site_inputs VALUES (?,?,?)',(uid,json.dumps(data.model_dump(mode='json',exclude_none=True)),now()))
    return dict(status='saved',source_type='Operator input; not independently verified')
@app.post('/api/v1/machine-inputs/{uid}')
def write_machine_inputs(uid:str,data:MachineInputs):
    get_mine(data.mine_id)
    with connection() as c:
        if not c.execute('SELECT id FROM machines WHERE id=?',(uid,)).fetchone():raise HTTPException(404,'Unknown benchmark record ID')
        c.execute('INSERT OR REPLACE INTO machine_inputs VALUES (?,?,?)',(uid,json.dumps(data.model_dump(mode='json')),now()))
    return dict(status='saved',warning='Asset mapping is operator supplied; sensor history remains the supplied benchmark, not live asset telemetry')
@app.post('/api/v1/shift-logs/{uid}')
def write_shift_log(uid:str,data:ShiftLog):
    get_mine(uid)
    with connection() as c:c.execute('INSERT OR REPLACE INTO shift_logs VALUES (?,?,?,?,?)',(uid,str(data.day),data.shift,json.dumps(data.model_dump(mode='json')),now()))
    return dict(status='saved',source_type='Operator-entered shift record')
@app.get('/api/v1/replay-dates')
def replay_dates():
    import pandas as pd
    d=pd.read_csv(ARTIFACTS/'production_history.csv')
    return dict(blocks=sorted(d.Mine_Block.unique()),dates=sorted(d.Date.unique()))


def resolve_options(uid,opts):
    from backend.store import site_inputs
    values=opts.model_dump(mode='json');stored=site_inputs(uid)
    for key,value in stored.items():
        if key in values and key not in opts.model_fields_set:values[key]=value
    return values


from datetime import datetime,timezone
from pydantic import model_validator
class SensorReading(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    node_id:str=Field(min_length=2,max_length=100)
    kind:Literal['rainfall_mm_h','soil_moisture_pct','pore_pressure_kpa','factor_of_safety','inflow_gpm','discharge_gpm','cycle_minutes','battery_pct','signal_dbm']
    value:float
    observed_at:datetime
    source:str=Field(min_length=5,max_length=500)
    operator:str=Field(min_length=2,max_length=100)
    latitude:float|None=Field(None,ge=-90,le=90)
    longitude:float|None=Field(None,ge=-180,le=180)
    screening_radius_m:float|None=Field(None,gt=0,le=5000)
    alert_above:float|None=None
    alert_below:float|None=None
    @model_validator(mode='after')
    def validate_measurement(self):
        if self.observed_at.tzinfo is None:raise ValueError('Observation time needs timezone')
        if self.observed_at>datetime.now(timezone.utc):raise ValueError('Observation time cannot be in the future')
        if (self.latitude is None)!=(self.longitude is None):raise ValueError('Supply both coordinates')
        if self.kind!='signal_dbm' and self.value<0:raise ValueError('Value must be nonnegative')
        if self.kind in ['soil_moisture_pct','battery_pct'] and self.value>100:raise ValueError('Percentage must be <=100')
        if self.alert_below is not None and self.alert_above is not None and self.alert_below>=self.alert_above:raise ValueError('Lower threshold must be below upper threshold')
        return self
@app.get('/api/v1/executive')
def executive_features():
    from backend.features import executive
    return executive()
@app.get('/api/v1/telemetry/{uid}')
def telemetry_features(uid:str,block:Literal['Block-A','Block-B','Block-C']='Block-A'):
    from backend.features import telemetry
    return telemetry(uid,block)
@app.post('/api/v1/measurements/{uid}')
def record_measurement(uid:str,data:SensorReading):
    get_mine(uid)
    with connection() as c:
        cur=c.execute('INSERT INTO sensor_readings(mine_id,payload,created_at) VALUES (?,?,?)',(uid,json.dumps(data.model_dump(mode='json')),now()))
    return dict(id=cur.lastrowid,status='recorded',source_type='Operator-supplied observation; not a live sensor connection')
@app.post('/api/v1/compare/{uid}')
def compare_features(uid:str,opts:Options):
    from backend.features import compare
    return compare(uid,resolve_options(uid,opts))
@app.get('/api/v1/outcomes')
def outcomes_features(mine_id:str|None=None):
    from backend.features import outcome_summary
    if mine_id:get_mine(mine_id)
    return outcome_summary(mine_id)
