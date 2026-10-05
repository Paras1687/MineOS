"""Conditional production scenarios and constrained alternatives; all assumptions exposed."""
import json,math,random,hashlib,time
from datetime import date,timedelta
import numpy as np,pandas as pd
from functools import lru_cache
from backend.config import ARTIFACTS
from backend.operations import artifact,fleet
from backend.catalog import catalog
from backend.weather import forecast as weather_forecast
from backend.store import distance,save_action,records,now,site_inputs,shift_logs

def get_mine(uid):
    for m in catalog():
        if m['id']==uid and m['target']==1:return m
    raise KeyError('Unknown occurrence')

def history_for(uid,selected_block=None):
    mine=get_mine(uid)
    if mine['country']!='India':raise ValueError('Operational demo is mapped only to India occurrences')
    block=site_inputs(uid).get('block') or selected_block or 'Block-A'
    d=pd.read_csv(ARTIFACTS/'production_history.csv');return d[d.Mine_Block==block].sort_values(['Date','Shift']),block

def predict(uid,opts):
    mine=get_mine(uid); h,block=history_for(uid,opts.get('replay_block'));
    if opts.get('mode')=='replay':return replay(uid,opts)
    inputs=site_inputs(uid);logs=shift_logs(uid)
    bundle=artifact('production.joblib'); equipment=fleet(uid)['machines']
    simulation=None
    if opts.get('simulate_events'):
        bucket=int(time.time()//300)
        rng=random.Random(hashlib.sha256(f'{uid}/{bucket}'.encode()).hexdigest())
        failed=[r['id'] for r in equipment if rng.random()<r['risk_score']]
        simulation=dict(snapshot=bucket,failed_machine_ids=failed,blast_delay_hours=rng.choice([0,.25,.5,1,1.5,2]),source='Simulated 5-minute snapshot; assigned mine sensors only. Blasting logs unavailable, delay is synthetic.')
        equipment=[{**r,'status':'inspect' if r['id'] in failed else r['status']} for r in equipment]
    available=100*np.mean([r['status']=='operational' for r in equipment]) if equipment else 0
    risk=100*np.mean([r['risk_score'] for r in equipment]) if equipment else 100
    if opts.get('fleet_available_pct_override') is not None:available=opts['fleet_available_pct_override']
    if opts.get('fleet_risk_pct_override') is not None:risk=opts['fleet_risk_pct_override']
    period=opts.get('period','daily'); ndays={'shift':1,'daily':1,'weekly':7,'monthly':30}[period]
    shifts=[opts.get('shift','Shift-1')] if period=='shift' else ['Shift-1','Shift-2','Shift-3']
    weather=weather_forecast(mine['latitude'],mine['longitude']) if opts.get('live_weather') else dict(status='not requested',days=[],retrieved_at=None)
    by_date={r['date']:r for r in weather['days']};rows=[]
    start=date.fromisoformat(opts.get('start_date') or date.today().isoformat())
    delay=opts.get('blast_delay_hours')
    if delay is None:delay=simulation['blast_delay_hours'] if simulation else 0.
    target_override=opts.get('target_per_shift_mt')
    price=opts.get('price_per_tonne',10000.); variable=opts.get('variable_cost_per_tonne',4000.)
    for day in range(ndays):
        dt=(start+timedelta(days=day)).isoformat()
        for shift in shifts:
            hist=h[h.Shift==shift]; last=hist.iloc[-1]; recent=hist.tail(7)
            conditions=by_date.get(dt);current=weather.get('current') or {}
            if conditions and current.get('time','')[:10]==dt:
                conditions={**conditions,'temperature_c':current.get('temperature_2m',conditions['temperature_c']),'humidity_pct':current.get('relative_humidity_2m',conditions['humidity_pct'])}
            rain=opts.get('rainfall_mm')
            rain=float(rain if rain is not None else (conditions['rainfall_mm'] if conditions else recent.Rainfall_mm.mean()))
            row={k:float(last[k]) for k in bundle['features']}
            row.update(lag1=float(last.Actual_Production_MT),rolling7=float(recent.Actual_Production_MT.mean()),
                Rainfall_mm=rain,Temperature_C=conditions['temperature_c'] if conditions else float(recent.Temperature_C.mean()),
                Humidity_pct=conditions['humidity_pct'] if conditions else float(recent.Humidity_pct.mean()),
                Predicted_Overall_Equipment_Availability_pct=float(available),Predicted_Average_Equipment_Failure_Risk_pct=float(risk))
            log=next((v for v in logs if v['day']==dt and v['shift']==shift),None)
            row['Planned_Target_MT']=float(target_override if target_override is not None else (log['target_mt'] if log else last.Planned_Target_MT))
            model_mt=max(0,float(bundle['model'].predict(pd.DataFrame([row])[bundle['features']])[0]))
            # Blast delay is an explicit engineering adjustment, absent from the training CSV.
            log=next((v for v in logs if v['day']==dt and v['shift']==shift),None)
            shift_delay=log['blast_delay_hours'] if log and opts.get('blast_delay_hours') is None else delay
            active=max(0,row['Planned_Working_Hours']-shift_delay-opts.get('equipment_downtime_hours',0))
            multiplier=active/max(.01,row['Planned_Working_Hours'])
            value=min(model_mt*multiplier,row['Planned_Target_MT']*1.25)
            if available==0:value=0
            radius=bundle['radius']*multiplier
            rows.append(dict(actual_mt=log['actual_mt'] if log else None,actual_source=log['source'] if log else None,blast_source=log['source'] if log and opts.get('blast_delay_hours') is None else 'Scenario assumption',date=dt,shift=shift,target_mt=row['Planned_Target_MT'],predicted_mt=round(value,2),
                lower_mt=round(max(0,value-radius),2),upper_mt=round(value+radius,2),
                rainfall_mm=rain,weather_source='user scenario' if opts.get('rainfall_mm') is not None else (weather['status'] if conditions else 'historical scenario proxy'),
                active_hours=round(active,2),planned_hours=row['Planned_Working_Hours']))
    # Approved plans alter forecast only, never historical actual production.
    approved=[]
    for action in (records() if opts.get('apply_approved',True) else []):
        if action['mine_id']!=uid or action['state']!='accepted':continue
        payload=action['payload'];schedule=payload.get('recovery_schedule',[]);applied=0
        for r in rows:
            planned=next((v['gain_mt'] for v in schedule if v['date']==r['date'] and v['shift']==r['shift']),0)
            gain=min(planned,max(0,r['target_mt']-r['predicted_mt']))
            r.setdefault('baseline_mt',r['predicted_mt']);r.setdefault('stockpile_supply_mt',0)
            if payload['kind']=='stockpile':r['stockpile_supply_mt']=round(gain,2)
            else:
                r['predicted_mt']=round(r['predicted_mt']+gain,2)
                r['lower_mt']=round(r['lower_mt']+gain,2);r['upper_mt']=round(r['upper_mt']+gain,2)
            applied+=gain
        if applied:approved.append(dict(id=action['id'],title=payload['title'],kind=payload['kind'],gain_mt=round(applied,2),cost_inr=payload['action_cost_inr'],machine_id=payload.get('machine_id')))
    total=lambda k:round(sum(r[k] for r in rows),2)
    predicted=total('predicted_mt');target=total('target_mt');stockpile=sum(r.get('stockpile_supply_mt',0) for r in rows);gap=max(0,target-predicted-stockpile)
    historical_rows=[dict(date=str(r.Date),shift=r.Shift,actual_mt=float(r.Actual_Production_MT),target_mt=float(r.Planned_Target_MT),predicted_mt=None) for r in h.tail(21).itertuples()]
    return dict(input_provenance=dict(mapping=inputs or {'source':'Unmapped dataset template'},costs='Operator input / editable scenario assumptions',forecast='Model estimate',events='Simulated' if simulation else 'Recorded logs where available; otherwise scenario assumptions'),updated_at=now(),simulation=simulation,approved_actions=approved,stockpile_supply_mt=round(stockpile,2),historical_rows=historical_rows,mine=mine,period=period,start_date=start.isoformat(),rows=rows,target_mt=target,predicted_mt=predicted,
        lower_mt=total('lower_mt'),upper_mt=total('upper_mt'),shortfall_mt=round(gap,2),shortfall_pct=round(gap/target*100,2) if target else 0,
        gross_revenue_at_risk_inr=round(gap*price,2),price_per_tonne=price,variable_cost_per_tonne=variable,
        current_production_mt=(round(sum(r['actual_mt'] for r in rows),2) if all(r.get('actual_mt') is not None for r in rows) else None),last_historical_production_mt=float(h[h.Date==h.Date.max()].Actual_Production_MT.sum()),
        historical_as_of=str(h.Date.max()),fleet_availability_pct=round(available,2),fleet_risk_pct=round(risk,2),weather=weather,
        assumptions=dict(blast_delay_hours=delay,blast_source='synthetic scenario; no mine blasting logs supplied',price_source='user-editable scenario, not current market price',
            production_mapping=(f'Operator-linked {block}: '+inputs['source']+' (not independently verified)' if inputs.get('block') else f'Unmapped {block} template selected for scenario. No mine-specific history established'),
            interval='Sum of shift error ranges; not a validated aggregate confidence interval'),
        warnings=['Current production is unavailable without a dated mine feed.',
                  'Realized-weather evaluation does not establish live-forecast accuracy.',
                  'Historical production/equipment-to-mine mappings are simulated.',
                  'Monthly dates beyond weather coverage use labelled historical weather scenarios.'] )

def alternatives(uid,opts):
    if opts.get('mode')=='replay':return dict(baseline=predict(uid,opts),actions=[],note='Historical replay: review recorded actuals against model reconstruction. Switch to Forecast / scenario to plan future recovery.')
    base=predict(uid,opts);gap=base['shortfall_mt'];price=base['price_per_tonne'];variable=base['variable_cost_per_tonne']
    hours=sum(r['active_hours'] for r in base['rows']);actions=[]
    if base['approved_actions']:return dict(baseline=base,actions=[],note='Approved plan is reflected in the forecast. Review or revoke it before choosing another alternative.')
    if gap<=0:return dict(baseline=base,actions=[],note='No predicted shortfall; no recovery action required')
    def add(kind,title,gain,cost,why,**extra):
        gain=round(min(gap,max(0,gain)),2);cost=round(max(0,cost),2)
        if gain<=0:return
        gross=round(gain*price,2);net=round(gross-gain*variable-cost,2)
        actions.append(dict(kind=kind,title=title,extra_supply_mt=gain,extra_mined_mt=0 if kind=='stockpile' else gain,
            remaining_shortfall_mt=round(gap-gain,2),action_cost_inr=cost,gross_value_inr=gross,
            variable_cost_inr=round(gain*variable,2),net_benefit_inr=net,benefit_cost_ratio=round((gross-gain*variable)/cost,2) if cost else None,
            recovery_schedule=[dict(date=r['date'],shift=r['shift'],gain_mt=round(gain*max(0,r['target_mt']-r['predicted_mt'])/max(gap,.01),4)) for r in base['rows']],reason=why,advisory=True,period=base['period'],start_date=base['start_date'],**extra))
    capacity=opts.get('processing_headroom_tph',20.)
    stock=opts.get('stockpile_mt',200.)
    add('stockpile','Draw available stockpile',min(stock,capacity*hours),opts.get('stockpile_handling_per_tonne',250.)*min(stock,capacity*hours,gap),
        'Inventory and spare processing throughput cap dispatch recovery. This is extra supply, not newly mined ore.',
        constraints={'stockpile_mt':stock,'spare_processing_tph':capacity})
    all_fleet=fleet()['machines'];mine=base['mine']
    candidates=[m for m in all_fleet if m['eligible'] and m['role']=='loader' and m['mine_id']!=uid]
    candidates=sorted(candidates,key=lambda m:distance(mine,m))[:6]
    for machine in candidates:
        km=distance(mine,machine)
        if km>opts.get('max_transfer_km',200.):continue
        travel=km/opts.get('transport_speed_kph',25.)+2
        useful=max(0,hours-travel)
        gain=min(machine['capacity_tph'],capacity)*useful*(1-machine['risk_score'])
        cost=opts.get('mobilization_inr',8000.)+km*opts.get('transport_inr_per_km',120.)
        add('transfer',f"Redeploy standby {machine['id']}",gain,cost,
            'Only unreserved, low-risk standby loaders are candidates. Transfer/setup time and spare plant capacity constrain recovery. Source allocated capacity is unchanged.',
            machine_id=machine['id'],source_mine=machine['mine_name'],distance_km=round(km,1),travel_hours=round(travel,1),constraints={'donor_production_loss_mt':0,'standby_verified_in_demo':True,'processing_headroom_tph':capacity})
    extra_hours=opts.get('extra_shift_hours',1.)*len(base['rows'])
    add('schedule','Extend permitted working window',capacity*extra_hours,extra_hours*opts.get('overtime_cost_per_hour',3000.),
        'Requires operator-approved working-time limits, labour and spare processing capacity. No automatic blast or safety-setting changes.',constraints={'extra_hours':extra_hours,'approval_required':True})
    actions.sort(key=lambda a:a['net_benefit_inr'],reverse=True)
    persisted=[dict(rank=i+1,**save_action(uid,a)) for i,a in enumerate(actions)]
    return dict(baseline=base,actions=persisted,note='Alternatives, not additive. Shared equipment, stockpile and processing capacity cannot be counted twice. All costs/capacities are editable demo assumptions.')


def replay(uid,opts):
    h,block=history_for(uid,opts.get('replay_block'));bundle=artifact('production.joblib')
    day=opts.get('replay_date') or str(h.Date.min());period=opts.get('period','daily')
    end=(date.fromisoformat(day)+timedelta(days={'shift':1,'daily':1,'weekly':7,'monthly':30}[period])).isoformat()
    selected=h[(h.Date>=day)&(h.Date<end)]
    if period=='shift':selected=selected[selected.Shift==opts.get('shift','Shift-1')]
    if selected.empty:raise ValueError('No supplied history for this replay date/block')
    values=np.maximum(0,bundle['model'].predict(selected[bundle['features']]))
    rows=[dict(date=str(r.Date),shift=r.Shift,actual_mt=float(r.Actual_Production_MT),target_mt=float(r.Planned_Target_MT),predicted_mt=round(float(v),2),rainfall_mm=float(r.Rainfall_mm),weather_source='Recorded dataset weather',lower_mt=round(float(max(0,v-bundle['radius'])),2),upper_mt=round(float(v+bundle['radius']),2)) for r,v in zip(selected.itertuples(),values)]
    total=lambda k:round(sum(r[k] for r in rows),2)
    target=total('target_mt');output=total('predicted_mt');gap=max(0,target-output)
    return dict(mode='replay',updated_at=now(),mine={**get_mine(uid),'name':block+' · historical dataset replay'},period=period,start_date=day,rows=rows,target_mt=target,predicted_mt=output,shortfall_mt=round(gap,2),shortfall_pct=round(gap/target*100,2) if target else 0,lower_mt=total('lower_mt'),upper_mt=total('upper_mt'),current_production_mt=total('actual_mt'),historical_rows=[],last_historical_production_mt=total('actual_mt'),historical_as_of=str(selected.Date.max()),fleet_availability_pct=float(selected.Predicted_Overall_Equipment_Availability_pct.mean()),fleet_risk_pct=float(selected.Predicted_Average_Equipment_Failure_Risk_pct.mean()),weather=dict(status='Recorded dataset weather; no live weather applied to past dates',retrieved_at=None,days=[]),simulation=None,approved_actions=[],stockpile_supply_mt=0,gross_revenue_at_risk_inr=round(gap*opts.get('price_per_tonne',10000),2),assumptions=dict(blast_delay_hours=None,production_mapping='Original '+block+' rows; not attributed to selected mine without documented mapping. Retrospective model reconstruction may include training rows; not a new holdout evaluation.',interval='Model error estimate',blast_source='No blast data in supplied history'),warnings=[])
