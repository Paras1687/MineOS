"""Point-grade evidence and explicit tonnage scenarios; no online training."""
import hashlib
import json
import threading
from functools import lru_cache
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.config import ARTIFACTS
from backend.grade_data import FEATURES, normalize, km

_lock = threading.RLock()
FILES = ['grade_reserve_metadata.json', 'grade_reserve_data.json', 'grade_spatial_v1.joblib']


class Bounds(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    lower: float = Field(gt=0, le=1e12)
    upper: float = Field(gt=0, le=1e12)

    @model_validator(mode='after')
    def ordered(self):
        if self.upper < self.lower:
            raise ValueError('Upper bound must be at least the lower bound.')
        return self


class TonnageScenario(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, str_strip_whitespace=True)
    area_m2: Bounds
    equivalent_thickness_m: Bounds
    bulk_density_t_m3: Bounds
    basis: str = Field(min_length=5, max_length=1000)

    @model_validator(mode='after')
    def plausible_units(self):
        if self.equivalent_thickness_m.upper > 10000 or self.bulk_density_t_m3.upper > 25:
            raise ValueError('Check units: thickness in metres (<=10000), bulk density in t/m3 (<=25).')
        return self


class LocalGeology(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    Geology: str = Field(min_length=2, max_length=500)
    Lithology: str = Field(min_length=2, max_length=500)
    Host_Rock: str = Field(min_length=2, max_length=500)
    Formation: str = Field(default='Unknown', max_length=500)
    Age: str = Field(default='Unknown', max_length=500)
    source_reference: str = Field(min_length=5, max_length=1000)


class EstimateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    prospectivity_score: float | None = Field(default=None, ge=0, le=1)
    geology: LocalGeology | None = None
    scenario: TonnageScenario | None = None


def signature():
    return tuple((ARTIFACTS/f).stat().st_mtime_ns for f in FILES)


@lru_cache(maxsize=2)
def load_bundle(sig):
    meta = json.loads((ARTIFACTS/FILES[0]).read_text(encoding='utf-8'))
    data = json.loads((ARTIFACTS/FILES[1]).read_text(encoding='utf-8'))
    if meta['version'] != data['version']:
        raise ValueError('Prepared data and model versions disagree; rerun offline preparation.')
    if hashlib.sha256((ARTIFACTS/FILES[2]).read_bytes()).hexdigest() != meta['model_sha256']:
        raise ValueError('Grade model checksum mismatch.')
    return meta, data['groups'], joblib.load(ARTIFACTS/FILES[2])


def versions():
    meta, _, _ = load_bundle(signature())
    return {k:meta[k] for k in ['version','data_sha256','model_sha256','source_filename']}


def tonnage(scenario, grade):
    x = scenario
    low = x['area_m2']['lower'] * x['equivalent_thickness_m']['lower'] * x['bulk_density_t_m3']['lower']
    high = x['area_m2']['upper'] * x['equivalent_thickness_m']['upper'] * x['bulk_density_t_m3']['upper']
    contained = None
    if grade.get('range_pct') and not grade.get('quality_flags'):
        lo, hi = grade['range_pct']
        contained = dict(lower=low*lo/100, upper=high*hi/100, unit='tonnes of contained Mn',
                         basis='Material tonnes × grade fraction; no recovery, dilution or economics applied.')
    return dict(status='calculated_from_assumptions', label='Tonnage scenario', lower=low, upper=high,
                unit='metric tonnes of mineralized material', inputs=x, formula='area_m2 × equivalent_thickness_m × bulk_density_t_m3',
                contained_mn=contained, limitations='Indicative geometry scenario, not an economically recoverable or proven reserve.')


def grade_result(point, known, groups, models, meta, evidence):
    if known:
        parsed = [r['grade'] for r in known['records'] if r['grade']['kind'] in ['exact','range','lower_bound']]
        if parsed:
            unique = {(g['kind'],g['lower_pct'],g['upper_pct']) for g in parsed}
            if len(unique) > 1:
                return dict(status='conflicting_records', method='Supplied Mn grade records', recorded_values=parsed,
                            reason='Duplicate-location records disagree; resolve source values before using a single grade.')
            g = parsed[0]
            return dict(status='recorded', method='Supplied Mn grade record', kind=g['kind'],
                        grade_pct=g['lower_pct'] if g['kind']=='exact' else None,
                        lower_pct=g['lower_pct'], upper_pct=g['upper_pct'],
                        range_pct=[g['lower_pct'],g['upper_pct']] if g['upper_pct'] is not None else None,
                        quality_flags=g.get('quality_flags',[]), recorded_values=parsed,
                        source_rows=[r['row'] for r in known['records']],
                        reason='Source labels are unverified; lower bounds remain bounds.')
    nearest = min(groups, key=lambda g:km(point,g))
    region = nearest['region']
    radius = 25 if region=='India' else 100
    gap = km(point,nearest)
    if gap > radius:
        return dict(status='insufficient_data', reason=f'Outside supported locations: nearest source is {gap:.1f} km away (limit {radius} km).',
                    next_input='Local assay or source-backed geology plus independently evaluated regional labels.')
    row = known['geology'] if known else ({f:normalize(evidence.get(f)) for f in FEATURES} if evidence else None)
    if not row:
        return dict(status='insufficient_data', reason='Coordinates and a CNN score do not establish local geology or grade.',
                    next_input='Supply local geology, lithology, host rock and a source reference, or obtain an assay.')
    training = [g for g in groups if g['region']==region and g['target'] is not None]
    support = [g for g in training if g['geology']['Geology']==row['Geology'] and row['Geology']!='unknown'
               and all(row[c]!='unknown' and g['geology'][c]==row[c] for c in ['Lithology','Host_Rock'])]
    evaluation = meta['evaluation'][region]
    if len(support) < 3 or any(row[f]!='unknown' and row[f] not in {g['geology'][f] for g in training} for f in FEATURES):
        return dict(status='insufficient_data', region=region, matching_locations=len(support),
                    reason='Unfamiliar geology or fewer than three independent matching locations.',
                    next_input='Add verified local assays and geology; run offline validation before estimating.')
    if not evaluation['prediction_enabled']:
        return dict(status='insufficient_data', region=region, evaluation={k:v for k,v in evaluation.items() if k!='folds'},
                    reason='Regional model did not meet the spatial evaluation gate (15 evaluated groups and lower MAE than median baseline).',
                    next_input='More spatially distinct, verified grade labels in this region.')
    value = float(models[region].predict(pd.DataFrame([row]))[0])
    radius = evaluation['screening_error_p90_pct']
    return dict(status='model_estimated', method='Regional geological-analogue random forest', region=region,
                grade_pct=round(value,2), range_pct=[round(max(0,value-radius),2),round(min(100,value+radius),2)],
                range_basis='P90 absolute error in spatial cross-validation; descriptive screening range, not confidence.',
                matching_locations=len(support), geology_source='Supplied site CSV' if known else evidence['source_reference'],
                evaluation={k:v for k,v in evaluation.items() if k!='folds'},
                reason='Conditional on mineralization. Predictions are compared with supplied interval midpoint labels.')


@lru_cache(maxsize=512)
def cached_estimate(sig, canonical):
    body = json.loads(canonical)
    meta, groups, models = load_bundle(sig)
    point = {k:body[k] for k in ['latitude','longitude']}
    nearest = min(groups,key=lambda g:min(km(point,r) for r in g['records']))
    # Coordinate association only; explicitly not a deposit extent.
    gap = min(km(point,r) for r in nearest['records'])
    known = nearest if gap <= .025 else None
    grade = grade_result(point,known,groups,models,meta,body.get('geology'))
    reserves = [dict(**r['reserve'],name=r['name'],csv_row=r['row'],source=r['source'])
                for r in known['records'] if r['reserve']] if known else []
    result = dict(location=point, prospectivity_score=body.get('prospectivity_score'),
                  prospectivity_basis='Displayed CNN index supplied by the calling workflow; not used to derive grade or tonnage.',
                  version=meta['version'], data_version=meta['data_sha256'], model_version=meta['model_sha256'],
                  sources=dict(filename=meta['source_filename'], original_path=meta['source_path'],
                               csv_rows=[r['row'] for r in known['records']] if known else [],
                               provenance='User-supplied CSV; provenance not independently verified'),
                  matched_location=dict(id=known['id'],distance_m=round(gap*1000,2),tolerance_m=25,
                                        names=[r['name'] for r in known['records']]) if known else None,
                  grade=grade,
                  reserve=dict(status='recorded' if reserves else 'insufficient_data', label='Supplied reserve record',
                               records=reserves, reason='Source figure retained in original units; no new reserve prediction.' if reserves
                               else 'No reserve record associated within 25 m of these coordinates.'),
                  tonnage=dict(status='insufficient_data',label='Indicative tonnage',
                               missing_inputs=['mineralized footprint area (m²)','equivalent thickness (m)','bulk density (t/m³)'],
                               reason='Source contains no verified complete site geometry/density. Generic depth/thickness descriptions are not used.'),
                  limitations=meta['limitations']+['A nearby match does not establish geological continuity or mineability.',
                                                   'Satellite patch area and 5 km scan area are never used as mineralized area.'])
    if body.get('scenario'):
        result['tonnage'] = tonnage(body['scenario'],grade)
    return result


def estimate(request):
    canonical = json.dumps(request.model_dump(mode='json'),sort_keys=True,separators=(',',':'))
    with _lock:
        # New model/data stat signature invalidates both the bundle and response cache.
        return json.loads(json.dumps(cached_estimate(signature(),canonical)))
