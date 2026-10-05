import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit,train_test_split
from sklearn.metrics import average_precision_score,recall_score,precision_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
from backend.config import DATA,ARTIFACTS,ROOT,MASTER
from backend.catalog import grade_range

SENSORS=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
PROD=['lag1','rolling7','Planned_Target_MT','Planned_Working_Hours','Mn_Grade_pct','Rainfall_mm','Temperature_C','Humidity_pct',
      'Predicted_Overall_Equipment_Availability_pct','Predicted_Average_Equipment_Failure_Risk_pct']

def main():
    reports={}
    tr=pd.read_csv(DATA/'ai4i2020_train_FINAL.csv'); te=pd.read_csv(DATA/'ai4i2020_test_FINAL.csv')
    overlap=set(tr.UDI)&set(te.UDI)
    if overlap: raise ValueError('Machine UDI overlap across train and test')
    m=RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=4)
    m.fit(tr[SENSORS],tr['Machine failure'])
    prob=m.predict_proba(te[SENSORS])[:,1]
    reports['health']=dict(test_rows=len(te),pr_auc=float(average_precision_score(te['Machine failure'],prob)),
        precision=float(precision_score(te['Machine failure'],prob>=.5,zero_division=0)),
        recall=float(recall_score(te['Machine failure'],prob>=.5)),features=SENSORS,
        limitations='Synthetic AI4I benchmark; current-condition classification, no validated failure lead time or RUL; mining machine types are supplied adaptations')
    joblib.dump(m,ARTIFACTS/'health.joblib')
    d=pd.concat([pd.read_csv(ROOT/'backend/models/production_forecast'/f'MineOS_Production_Intelligence_{s}.csv') for s in ['Train','Test']],ignore_index=True)
    d['Date']=pd.to_datetime(d.Date); d=d.sort_values(['Mine_Block','Shift','Date'])
    g=d.groupby(['Mine_Block','Shift'])['Actual_Production_MT']
    d['lag1']=g.shift(1)
    d['rolling7']=g.transform(lambda z:z.shift(1).rolling(7,min_periods=3).mean())
    d=d.dropna(subset=PROD)
    train=d[d.Date<'2025-06-01']; cal=d[(d.Date>='2025-06-01')&(d.Date<'2025-08-07')]; test=d[d.Date>='2025-08-07']
    model=XGBRegressor(n_estimators=220,max_depth=3,learning_rate=.045,subsample=.9,colsample_bytree=.9,reg_lambda=5,random_state=42,n_jobs=4)
    model.fit(train[PROD],train.Actual_Production_MT)
    residual=np.abs(model.predict(cal[PROD])-cal.Actual_Production_MT)
    radius=float(np.quantile(residual,min(1,np.ceil((len(residual)+1)*.9)/len(residual)),method='higher'))
    pred=model.predict(test[PROD]); actual=test.Actual_Production_MT
    reports['production']=dict(test_rows=len(test),calibration_rows=len(cal),train_end=str(train.Date.max().date()),test_start=str(test.Date.min().date()),
        mae_mt=float(mean_absolute_error(actual,pred)),rmse_mt=float(np.sqrt(mean_squared_error(actual,pred))),
        rolling7_mae_mt=float(mean_absolute_error(actual,test.rolling7)),interval_radius_mt=radius,
        empirical_coverage=float(np.mean(np.abs(actual-pred)<=radius)),features=PROD,
        limitations='Conditional shift-output model evaluated with realized weather/equipment conditions, not archived weather forecasts. Multi-day and mine transfer are unvalidated scenarios. Historical source mine IDs are not present.')
    joblib.dump(dict(model=model,features=PROD,radius=radius),ARTIFACTS/'production.joblib')
    d.to_csv(ARTIFACTS/'production_history.csv',index=False)
    # Grade: positive records only, no reserve/grade/subsurface-grade or synthetic negative zeros as features.
    geo=pd.read_csv(MASTER); geo=geo[geo.Target==1].copy()
    geo['label']=geo.Grade.map(grade_range); geo=geo[geo.label.notna()].copy()
    geo['grade_mid']=geo.label.map(lambda z:sum(z)/2)
    cols=['Geology','Lithology','Host_Rock','Formation','Age']
    geo[cols]=geo[cols].fillna('Unknown').astype(str)
    groups=(pd.to_numeric(geo.Latitude,errors='coerce').fillna(0).floordiv(1).astype(str)+'/'+pd.to_numeric(geo.Longitude,errors='coerce').fillna(0).floordiv(1).astype(str))
    a,b=next(GroupShuffleSplit(test_size=.25,random_state=42).split(geo,groups=groups))
    enc=ColumnTransformer([('geo',OneHotEncoder(handle_unknown='ignore'),cols)])
    reg=Pipeline([('encode',enc),('model',RandomForestRegressor(n_estimators=160,min_samples_leaf=5,max_depth=8,random_state=42,n_jobs=4))])
    reg.fit(geo.iloc[a][cols],geo.iloc[a].grade_mid)
    errors=np.abs(reg.predict(geo.iloc[b][cols])-geo.iloc[b].grade_mid)
    reports['grade']=dict(training_rows=len(a),evaluation_rows=len(b),mae_pct=float(errors.mean()),
        screening_error_pct=float(np.quantile(errors,.9)),status='Experimental analogue grade only; source ranges unverified; no field-calibrated confidence interval',features=cols)
    joblib.dump(dict(model=reg,features=cols,error=float(np.quantile(errors,.9)),vocabulary={c:sorted(geo.iloc[a][c].unique()) for c in cols}),ARTIFACTS/'grade.joblib')
    reports['resource']=dict(status='Blocked: unit and provenance verification required',reason='Source tons and unitless numbers cannot be treated as measured metric reserves. Borehole rows mostly generic surface/region descriptions. No defensible site-volume or recoverable reserve labels supplied.')
    (ARTIFACTS/'tabular_metrics.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2),flush=True)
if __name__=='__main__':main()
