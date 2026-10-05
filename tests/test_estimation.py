import unittest
import hashlib
from pathlib import Path
from pydantic import ValidationError
from backend.grade_data import parse_grade, parse_reserve, FEATURES
from backend.estimation import EstimateRequest, estimate, load_bundle, signature, TonnageScenario, tonnage, cached_estimate

class EstimationTests(unittest.TestCase):
    def test_parsing(self):
        for text in ['35-46% Mn; 0.2% P','35–46% Mn','Mn 35 to 46%, phosphorus 0.2%']:
            g=parse_grade(text);self.assertEqual(g['midpoint_pct'],40.5)
        for text in ['40% Mn, 0.2% P','0.2% phosphorus; 40% Mn','Mn 40% P 0.2%']:
            self.assertEqual(parse_grade(text)['midpoint_pct'],40)
        g=parse_grade('+48%');self.assertEqual(g['kind'],'lower_bound');self.assertIsNone(g['midpoint_pct'])
        for text in ['', 'N/A',None]:self.assertEqual(parse_grade(text)['kind'],'missing')
        self.assertEqual(parse_grade('70% MnO2')['kind'],'unresolved')
        self.assertEqual(parse_grade('50-20%')['kind'],'unresolved')
        self.assertIsNone(parse_reserve('6210000')['metric_tonnes'])
        self.assertEqual(parse_reserve('100 tons')['original_unit'],'tons')

    def test_geometry(self):
        data=dict(area_m2=dict(lower=100,upper=200),equivalent_thickness_m=dict(lower=2,upper=3),bulk_density_t_m3=dict(lower=3,upper=4),basis='Explicit test geometry')
        out=tonnage(TonnageScenario(**data).model_dump(),dict(range_pct=[40,50]))
        self.assertEqual((out['lower'],out['upper']),(600,2400));self.assertEqual(out['contained_mn']['upper'],1200)
        for bad in [{**data,'area_m2':dict(lower=0,upper=1)},{**data,'area_m2':dict(lower=2,upper=1)},{**data,'bulk_density_t_m3':dict(lower=30,upper=40)},{**data,'basis':'     '}]:
            with self.assertRaises(ValidationError):TonnageScenario(**bad)
        with self.assertRaises(ValidationError):TonnageScenario(basis='No geometry')
        with self.assertRaises(ValidationError):EstimateRequest(latitude=float('nan'),longitude=80)

    def test_known_unknown_and_cache(self):
        _,groups,_=load_bundle(signature());g=next(g for g in groups if g['region']=='India' and g['records'][0]['grade']['kind']=='lower_bound');r=g['records'][0]
        req=EstimateRequest(latitude=r['latitude'],longitude=r['longitude'],prospectivity_score=.95)
        a=estimate(req);self.assertEqual(a['grade']['status'],'recorded');self.assertIsNone(a['grade']['upper_pct']);self.assertEqual(a['reserve']['status'],'recorded')
        self.assertTrue(all(x['metric_tonnes'] is None for x in a['reserve']['records']))
        before=cached_estimate.cache_info().hits;self.assertEqual(a,estimate(req));self.assertGreater(cached_estimate.cache_info().hits,before)
        b=estimate(EstimateRequest(latitude=0,longitude=0,prospectivity_score=.99));self.assertEqual(b['grade']['status'],'insufficient_data');self.assertFalse(b['reserve']['records']);self.assertEqual(len(b['tonnage']['missing_inputs']),3)
        c=estimate(EstimateRequest(latitude=r['latitude']+.01,longitude=r['longitude']));self.assertEqual(c['grade']['status'],'insufficient_data')

    def test_spatial_audit_and_source(self):
        meta,groups,_=load_bundle(signature())
        self.assertEqual(meta['audit']['rows'],486);self.assertEqual(meta['audit']['region_location_groups']['India'],25)
        self.assertNotIn('Subsurface_Grade',FEATURES)
        for evaluation in meta['evaluation'].values():
            for fold in evaluation['folds']:
                if fold['status']=='evaluated':
                    self.assertFalse(set(fold['train_ids'])&set(fold['test_ids']));self.assertGreaterEqual(fold['min_distance_km'],10)
        rows=[r['row'] for g in groups for r in g['records']];self.assertEqual(len(rows),len(set(rows)))
        path=Path(meta['source_path'])
        if path.exists():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),meta['data_sha256'])

if __name__=='__main__':unittest.main()
