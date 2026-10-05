import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from backend import store
from backend.server import app
from backend.config import ARTIFACTS
from backend.catalog import grade_range
from backend.operations import health_probabilities

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.old=store.DB;store.DB=Path(cls.tmp.name)/'test.sqlite3'
        health_probabilities.cache_clear();cls.client=TestClient(app);cls.client.__enter__()
        cls.mine=next(m['id'] for m in cls.client.get('/api/v1/mines').json()['data'] if m['country']=='India')
    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None);store.DB=cls.old;health_probabilities.cache_clear();cls.tmp.cleanup()
    def test_reference_features(self):
        c=self.client;uid=self.mine
        r=c.get('/api/v1/executive');self.assertEqual(r.status_code,200);self.assertGreater(len(r.json()['sites']),0)
        r=c.get('/api/v1/telemetry/'+uid);self.assertEqual(r.status_code,200);self.assertEqual(len(r.json()['series']),90)
        body=dict(node_id='TEST-NODE',kind='soil_moisture_pct',value=45,observed_at='2025-01-01T00:00:00Z',source='Test instrument log',operator='Tester',alert_above=40)
        self.assertEqual(c.post('/api/v1/measurements/'+uid,json={**body,'value':101}).status_code,422)
        self.assertEqual(c.post('/api/v1/measurements/'+uid,json=body).status_code,200)
        self.assertTrue(c.get('/api/v1/telemetry/'+uid).json()['measurements'][0]['alert'])
        before=len(store.records())
        r=c.post('/api/v1/compare/'+uid,json=dict(period='shift',live_weather=False,rainfall_mm=45,blast_delay_hours=2,equipment_downtime_hours=1))
        self.assertEqual(r.status_code,200);d=r.json();self.assertEqual(len(d['drivers']),3);self.assertFalse(d['active']['approved_actions']);self.assertEqual(len(store.records()),before)
        self.assertEqual(c.get('/api/v1/outcomes').status_code,200)

    def test_health_and_coverage(self):
        d=self.client.get('/api/v1/health').json();self.assertTrue(all(d['models'].values()))
        rows=json.loads((ARTIFACTS/'raster_manifest.json').read_text());groups={}
        for r in rows:groups.setdefault(r['group'],set()).add(r['split'])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        self.assertEqual(len(rows),json.loads((ARTIFACTS/'raster_audit.json').read_text())['accepted'])
        if (ARTIFACTS/'patches.npy').exists():
            self.assertEqual(len(rows),len(np.load(ARTIFACTS/'patches.npy',mmap_mode='r')))
    def test_mine_filter(self):
        mines=self.client.get('/api/v1/mines').json()['data']
        self.assertEqual(len(mines),len({m['id'] for m in mines}))
        r=self.client.get('/api/v1/operational/equipment',params={'mine_id':self.mine});self.assertEqual(r.status_code,200)
        self.assertTrue(all(m['mine_id']==self.mine for m in r.json()['machines']))
        self.assertTrue(all(not m['location_live'] and m['rul_days'] is None for m in r.json()['machines']))
    def test_units_and_parsing(self):
        self.assertEqual(grade_range('Mn: 37.94% and P (Phosphorus): 0.24%'),[37.94,37.94])
        self.assertIsNone(grade_range('+48% Mn'));self.assertIsNone(grade_range('70-75% MnO2'))
        self.assertIsNone(self.client.get(f'/api/v1/mines/{self.mine}/geology').json()['resource_prediction'])
    def test_production_nonnegative_reconciled(self):
        for period,n in [('shift',1),('daily',3),('weekly',21),('monthly',90)]:
            r=self.client.post(f'/api/v1/production/{self.mine}',json={'period':period});self.assertEqual(r.status_code,200,r.text)
            d=r.json();self.assertEqual(len(d['rows']),n);self.assertIsNone(d['current_production_mt'])
            self.assertAlmostEqual(d['predicted_mt'],sum(x['predicted_mt'] for x in d['rows']),places=1)
            self.assertAlmostEqual(d['shortfall_mt'],max(0,d['target_mt']-d['predicted_mt']),places=1)
    def test_constraints_and_economics(self):
        opts={'target_per_shift_mt':1500,'price_per_tonne':9000,'variable_cost_per_tonne':4000}
        d=self.client.post(f'/api/v1/decisions/{self.mine}',json=opts).json();self.assertGreater(len(d['actions']),0)
        for a in d['actions']:
            self.assertLessEqual(a['extra_supply_mt'],d['baseline']['shortfall_mt'])
            self.assertAlmostEqual(a['net_benefit_inr'],a['extra_supply_mt']*5000-a['action_cost_inr'],places=1)
        closed=self.client.post(f'/api/v1/decisions/{self.mine}',json={**opts,'processing_headroom_tph':0,'stockpile_mt':0,'extra_shift_hours':0}).json()
        self.assertEqual(closed['actions'],[])
    def test_persistent_action_state(self):
        d=self.client.post(f'/api/v1/decisions/{self.mine}',json={'target_per_shift_mt':1500}).json()
        uid=d['actions'][0]['id'];body={'state':'accepted','actor':'Test operator','reason':'Integration test only'}
        self.assertEqual(self.client.post('/api/v1/actions/'+uid,json=body).status_code,200)
        other=d['actions'][1]['id']
        self.assertEqual(self.client.post('/api/v1/actions/'+other,json=body).status_code,422)
        self.assertEqual(self.client.post('/api/v1/actions/'+uid,json={**body,'state':'completed'}).status_code,422)
        body.update(state='completed',actual_recovered_mt=12,actual_cost_inr=3000)
        self.assertEqual(self.client.post('/api/v1/actions/'+uid,json=body).status_code,200)
        with store.connection() as c:self.assertEqual(c.execute('SELECT state FROM actions WHERE id=?',(uid,)).fetchone()[0],'completed')
        self.assertEqual(self.client.post('/api/v1/actions/'+uid,json={**body,'state':'accepted'}).status_code,422)
    def test_connected_approval_forecast(self):
        from unittest.mock import patch
        from backend import planning
        opts={'target_per_shift_mt':1500,'blast_delay_hours':3,'simulate_events':True}
        with patch('backend.planning.time.time',return_value=1800000000):
            base=self.client.post(f'/api/v1/production/{self.mine}',json=opts).json()
            decisions=self.client.post(f'/api/v1/decisions/{self.mine}',json=opts).json()
            a=next(a for a in decisions['actions'] if a['kind']=='schedule')
            body={'actor':'Test operator','reason':'Test connected planning','state':'acknowledged'}
            self.assertEqual(self.client.post('/api/v1/actions/'+a['id'],json=body).status_code,200)
            unchanged=self.client.post(f'/api/v1/production/{self.mine}',json=opts).json()
            self.assertEqual(base['predicted_mt'],unchanged['predicted_mt'])
            body['state']='accepted';self.assertEqual(self.client.post('/api/v1/actions/'+a['id'],json=body).status_code,200)
            after=self.client.post(f'/api/v1/production/{self.mine}',json=opts).json()
            self.assertGreater(after['predicted_mt'],base['predicted_mt'])
            self.assertLess(after['shortfall_mt'],base['shortfall_mt'])
            self.assertIsNone(after['current_production_mt'])
            again=self.client.post(f'/api/v1/production/{self.mine}',json=opts).json()
            self.assertEqual(after['predicted_mt'],again['predicted_mt'])
            ids={m['id'] for m in self.client.get('/api/v1/operational/equipment',params={'mine_id':self.mine}).json()['machines']}
            self.assertTrue(set(after['simulation']['failed_machine_ids'])<=ids)
            body['state']='rejected';self.client.post('/api/v1/actions/'+a['id'],json=body)
            reset=self.client.post(f'/api/v1/production/{self.mine}',json=opts).json()
            self.assertEqual(reset['predicted_mt'],base['predicted_mt'])
            delayed=self.client.post(f'/api/v1/production/{self.mine}',json={**opts,'blast_delay_hours':6}).json()
            self.assertLess(delayed['predicted_mt'],base['predicted_mt'])

    def test_source_records_and_replay(self):
        source={'source':'Test register only','operator':'Test operator','observed_at':'2026-09-29'}
        url='/api/v1/site-inputs/'+self.mine
        self.assertEqual(self.client.post(url,json={**source,'block':'Block-B','price_per_tonne':12345}).status_code,200)
        self.assertEqual(self.client.get(url).json()['inputs']['block'],'Block-B')
        replay=self.client.post('/api/v1/production/'+self.mine,json={'mode':'replay','replay_date':'2024-01-04'}).json()
        self.assertEqual(replay['mode'],'replay');self.assertIn('Block-B',replay['mine']['name'])
        self.assertEqual(replay['current_production_mt'],round(sum(r['actual_mt'] for r in replay['rows']),2))
        self.assertIsNone(replay['simulation'])
        day='2026-09-29'
        log={**source,'day':day,'shift':'Shift-1','actual_mt':88,'target_mt':120,'blast_delay_hours':2}
        self.assertEqual(self.client.post('/api/v1/shift-logs/'+self.mine,json=log).status_code,200)
        r=self.client.post('/api/v1/production/'+self.mine,json={'period':'shift','start_date':day}).json()
        self.assertEqual(r['current_production_mt'],88);self.assertEqual(r['target_mt'],120);self.assertEqual(r['price_per_tonne'],12345)
        with store.connection() as c:
            c.execute('DELETE FROM site_inputs WHERE mine_id=?',(self.mine,))
            c.execute('DELETE FROM shift_logs WHERE mine_id=?',(self.mine,))

    def test_validation_and_missing(self):
        self.assertEqual(self.client.post(f'/api/v1/production/{self.mine}',json={'price_per_tonne':-1}).status_code,422)
        self.assertEqual(self.client.get('/api/v1/operational/equipment?mine_id=invalid').status_code,404)
        self.assertEqual(self.client.post('/api/v1/exploration/predict',json={'longitude':181,'latitude':0}).status_code,422)
    def test_local_inference(self):
        from backend.exploration import predict_point
        rows=json.loads((ARTIFACTS/'raster_manifest.json').read_text())
        row=next(r for r in rows if r['split']=='test' and Path(r['path']).exists())
        r=predict_point(row['longitude'],row['latitude']);self.assertGreaterEqual(r['prospectivity_score'],0);self.assertLessEqual(r['prospectivity_score'],1)
        self.assertTrue(r['heatmap'].startswith('data:image/png;base64,'));self.assertFalse(r['field_validated'])
        self.assertIn('not a probability',r['score_status'])
        self.assertEqual(r['geographic_support']['radius_km'],25)
        train=next(r for r in rows if r['split']=='train' and Path(r['path']).exists())
        in_sample=predict_point(train['longitude'],train['latitude'])
        self.assertFalse(in_sample['sample_independent'])
        self.assertIn('IN-SAMPLE ONLY',in_sample['sample_label_note'])
    def test_world_fusion_preserves_training_footprint(self):
        from backend.models.geo_intelligence.world_fusion import prepare_image
        rng=np.random.default_rng(7)
        large=rng.uniform(100,4000,(9,160,180)).astype('float32')
        y0,x0=(160-64)//2,(180-64)//2
        self.assertTrue(np.allclose(prepare_image(large),prepare_image(large[:,y0:y0+64,x0:x0+64])))

if __name__=='__main__':unittest.main(verbosity=2)
