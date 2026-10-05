import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,precision_score,recall_score,roc_auc_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
from backend.config import DATA,ARTIFACTS,ROOT
from scripts.train_tabular import SENSORS,PROD

def main():
    report=json.loads((ARTIFACTS/'tabular_metrics.json').read_text())
    train=pd.read_csv(DATA/'ai4i2020_train_FINAL.csv');test=pd.read_csv(DATA/'ai4i2020_test_FINAL.csv')
    ti,vi=train_test_split(np.arange(len(train)),test_size=.2,random_state=42,stratify=train['Machine failure'])
    grids=[dict(n_estimators=160,max_depth=10,min_samples_leaf=3,class_weight='balanced_subsample'),
           dict(n_estimators=320,max_depth=8,min_samples_leaf=5,class_weight='balanced_subsample'),
           dict(n_estimators=240,max_depth=12,min_samples_leaf=8,class_weight='balanced_subsample')]
    candidates=[]
    for params in grids:
        model=RandomForestClassifier(**params,random_state=42,n_jobs=4);model.fit(train.iloc[ti][SENSORS],train.iloc[ti]['Machine failure'])
        p=model.predict_proba(train.iloc[vi][SENSORS])[:,1]
        candidates.append((average_precision_score(train.iloc[vi]['Machine failure'],p),params))
    _,hp=max(candidates,key=lambda x:x[0]);health=RandomForestClassifier(**hp,random_state=42,n_jobs=4)
    health.fit(train[SENSORS],train['Machine failure']);p=health.predict_proba(test[SENSORS])[:,1];yt=test['Machine failure'].to_numpy()
    report['health']=dict(test_rows=len(test),pr_auc=float(average_precision_score(yt,p)),roc_auc=float(roc_auc_score(yt,p)),
        precision_at_0_5=float(precision_score(yt,p>=.5,zero_division=0)),recall_at_0_5=float(recall_score(yt,p>=.5)),
        precision_at_0_15=float(precision_score(yt,p>=.15,zero_division=0)),recall_at_0_15=float(recall_score(yt,p>=.15)),
        selected_params=hp,selection='PR-AUC on stratified training-only validation split',features=SENSORS,
        limitations='AI4I synthetic machine benchmark; no failure lead time/RUL and no verified mapping to MOIL machinery.')
    joblib.dump(health,ARTIFACTS/'health.joblib')
    path=ROOT/'backend/models/production_forecast'
    d=pd.concat([pd.read_csv(path/f'MineOS_Production_Intelligence_{s}.csv') for s in ['Train','Test']],ignore_index=True)
    d['Date']=pd.to_datetime(d.Date);d=d.sort_values(['Mine_Block','Shift','Date']);g=d.groupby(['Mine_Block','Shift'])['Actual_Production_MT']
    d['lag1']=g.shift(1);d['rolling7']=g.transform(lambda z:z.shift(1).rolling(7,min_periods=3).mean());d=d.dropna(subset=PROD)
    tr=d[d.Date<'2025-06-01'];va=d[(d.Date>='2025-06-01')&(d.Date<'2025-08-07')];te=d[d.Date>='2025-08-07']
    signs={'Planned_Target_MT':1,'Planned_Working_Hours':1,'Rainfall_mm':-1,'Predicted_Overall_Equipment_Availability_pct':1,'Predicted_Average_Equipment_Failure_Risk_pct':-1}
    params=[dict(n_estimators=220,max_depth=3,learning_rate=.045,subsample=.9,colsample_bytree=.9,reg_lambda=5),
            dict(n_estimators=300,max_depth=2,learning_rate=.035,subsample=.85,colsample_bytree=.85,reg_lambda=10,min_child_weight=4),
            dict(n_estimators=240,max_depth=2,learning_rate=.04,subsample=.9,colsample_bytree=.9,reg_lambda=8,min_child_weight=5,monotone_constraints=tuple(signs.get(c,0) for c in PROD))]
    choices=[]
    for cfg in params:
        m=XGBRegressor(**cfg,random_state=42,n_jobs=4);m.fit(tr[PROD],tr.Actual_Production_MT)
        choices.append((mean_absolute_error(va.Actual_Production_MT,m.predict(va[PROD])),cfg,m))
    va_mae,pp,model=min(choices,key=lambda a:a[0]);vp=model.predict(va[PROD]);res=np.abs(vp-va.Actual_Production_MT.to_numpy())
    radius=float(np.quantile(res,min(1,np.ceil((len(res)+1)*.9)/len(res)),method='higher'))
    pred=model.predict(te[PROD]);actual=te.Actual_Production_MT.to_numpy()
    report['production']=dict(test_rows=len(te),validation_rows=len(va),calibration_rows=len(va),train_end=str(tr.Date.max().date()),
        test_start=str(te.Date.min().date()),mae_mt=float(mean_absolute_error(actual,pred)),rmse_mt=float(np.sqrt(mean_squared_error(actual,pred))),
        rolling7_mae_mt=float(mean_absolute_error(actual,te.rolling7)),validation_mae_mt=float(va_mae),
        interval_radius_mt=radius,empirical_coverage=float(np.mean(np.abs(actual-pred)<=radius)),selected_params={k:v for k,v in pp.items() if k!='monotone_constraints'},
        monotone_constraints=signs if 'monotone_constraints' in pp else {},selection='Lowest MAE on the 2025-06-01 to 2025-08-06 time validation window',features=PROD,
        limitations='Time-ordered evaluation, source mine IDs absent; historical weather is not a forecast; transfer across mines is unvalidated.')
    joblib.dump(dict(model=model,features=PROD,radius=radius),ARTIFACTS/'production.joblib')
    d.to_csv(ARTIFACTS/'production_history.csv',index=False)
    (ARTIFACTS/'tabular_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'health':report['health'],'production':report['production']},indent=2),flush=True)

if __name__=='__main__':main()
