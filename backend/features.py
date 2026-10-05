"""Traceable executive, telemetry and what-if features; no fabricated telemetry."""
import json,math
from datetime import date,datetime
import numpy as np
from backend.catalog import catalog
from backend.store import connection,now,site_inputs,shift_logs,records
from backend.operations import fleet
from backend.planning import get_mine,history_for,predict
from backend.exploration import grade_for


def readings(uid):
    with connection() as c:rows=c.execute('SELECT id,payload,created_at FROM sensor_readings WHERE mine_id=? ORDER BY id DESC LIMIT 300',(uid,)).fetchall()
    result=[]
    for row in rows:
        d=json.loads(row['payload']);d.update(id=row['id'],created_at=row['created_at'])
        d['alert']=(d.get('alert_above') is not None and d['value']>d['alert_above']) or (d.get('alert_below') is not None and d['value']<d['alert_below'])
        result.append(d)
    return result


def executive():
    machines=fleet()['machines'];sites=[];today=date.today().isoformat()
    for m in catalog():
        if m['target']!=1 or m['country']!='India':continue
        inputs=site_inputs(m['id']);logs=shift_logs(m['id']);current=[r for r in logs if r['day']==today]
        fm=[r for r in machines if r['mine_id']==m['id']];inspections=sum(r['status']=='inspect' for r in fm)
        actual=sum(r['actual_mt'] for r in current) if current else None
        target=sum(r['target_mt'] for r in current) if current else None
        gap=max(0,target-actual) if current else None
        sensors=readings(m['id']);latest={}
        for r in sensors:
            if r['kind'] not in latest or datetime.fromisoformat(r['observed_at'])>datetime.fromisoformat(latest[r['kind']]['observed_at']):latest[r['kind']]=r
        alerts=[r for r in latest.values() if r['alert']]
        sites.append(dict(id=m['id'],name=m['name'],latitude=m['latitude'],longitude=m['longitude'],mine_type=inputs.get('mine_type','Unknown'),depth_m=inputs.get('depth_m'),actual_mt=actual,target_mt=target,gap_mt=gap,reported_shifts=len(current),as_of=today if current else None,machines=len(fm),inspect=inspections,availability_pct=round(100*(len(fm)-inspections)/len(fm),1) if fm else None,mean_health_risk=round(float(np.mean([r['risk_score'] for r in fm])),4) if fm else None,grade_reported=m['reported_grade'],mapping_source=inputs.get('source','No documented mine/block mapping'),block=inputs.get('block'),telemetry=latest,alerts=alerts,attention=bool(inspections or alerts or (gap or 0)>0)))
    reporting=[s for s in sites if s['actual_mt'] is not None]
    return dict(sites=sites,updated_at=now(),totals=dict(actual_mt=round(sum(s['actual_mt'] for s in reporting),2),target_mt=round(sum(s['target_mt'] for s in reporting),2),reporting_sites=len(reporting),total_sites=len(sites),inspect=sum(s['inspect'] for s in sites),sensor_alerts=sum(len(s['alerts']) for s in sites)),note='Today totals cover entered shift logs only; missing shifts are not zero. Fleet health uses benchmark sensors; assignments may be simulated. Attention is a rule, not a calibrated mine-risk probability.')


def telemetry(uid,block='Block-A'):
    mine=get_mine(uid);h,block=history_for(uid,block);h=h.tail(90)
    series=[dict(date=str(r.Date),shift=r.Shift,rainfall_mm=float(r.Rainfall_mm),actual_mt=float(r.Actual_Production_MT),target_mt=float(r.Planned_Target_MT),availability_pct=float(r.Predicted_Overall_Equipment_Availability_pct)) for r in h.itertuples()]
    correlation=float(np.corrcoef(h.Rainfall_mm,h.Actual_Production_MT)[0,1]) if h.Rainfall_mm.std()>0 and h.Actual_Production_MT.std()>0 else None
    rs=readings(uid)
    return dict(mine=mine,block=block,series=series,correlation=correlation if correlation is not None and math.isfinite(correlation) else None,measurements=rs,fleet=fleet(uid),grade=grade_for(mine),source='Supplied '+block+' history: chronological dataset replay; original block, not live mine telemetry. Correlation is descriptive, not causal.',updated_at=now())


def compare(uid,options):
    opts={**options,'mode':'forecast','apply_approved':False,'simulate_events':False}
    active=predict(uid,opts)
    baseline=predict(uid,{**opts,'rainfall_mm':0,'blast_delay_hours':0,'equipment_downtime_hours':0,'fleet_available_pct_override':100,'fleet_risk_pct_override':0})
    drivers=[]
    for label,overrides in [('Rainfall set to zero',{'rainfall_mm':0}),('Blasting delay removed',{'blast_delay_hours':0}),('Fleet available; no downtime',{'fleet_available_pct_override':100,'fleet_risk_pct_override':0,'equipment_downtime_hours':0})]:
        alternative=predict(uid,{**opts,**overrides})
        drivers.append(dict(label=label,delta_mt=round(alternative['predicted_mt']-active['predicted_mt'],2)))
    return dict(active=active,baseline=baseline,delta_mt=round(active['predicted_mt']-baseline['predicted_mt'],2),drivers=drivers,method='One-factor scenario sensitivity using the production model plus explicit delay adjustments. Not SHAP, causal attribution or additive contributions. Approved plans and random events excluded for a stable comparison.')


def outcome_summary(uid=None):
    rows=[r for r in records() if not uid or r['mine_id']==uid];completed=[r for r in rows if r['state']=='completed']
    actual=sum(r['actual_recovered_mt'] or 0 for r in completed);cost=sum(r['actual_cost'] or 0 for r in completed)
    valued=[r for r in completed if r['payload'].get('extra_supply_mt',0)>0]
    gross=sum((r['actual_recovered_mt'] or 0)*r['payload'].get('gross_value_inr',0)/r['payload']['extra_supply_mt'] for r in valued)
    variable=sum((r['actual_recovered_mt'] or 0)*r['payload'].get('variable_cost_inr',0)/r['payload']['extra_supply_mt'] for r in valued)
    return dict(rows=rows,summary=dict(completed=len(completed),actual_recovery_mt=round(actual,2),actual_cost_inr=round(cost,2),estimated_gross_value_inr=round(gross,2),estimated_net_value_inr=round(gross-variable-cost,2),estimated_roi=(gross-variable-cost)/cost if cost else None),note='Recovery and action costs are operator entries. Value/ROI uses the saved scenario unit rates, not audited profit. Alternative recommendations are not summed as realized recovery. Register covers the latest 500 saved actions.')
