from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from backend.config import ARTIFACTS


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class WorldFusionCNN(nn.Module):
    def __init__(self, gravity_mean=0.0, gravity_variance=1.0, gravity_dropout_after_norm=False):
        super().__init__()
        self.gravity_dropout_after_norm = gravity_dropout_after_norm
        self.conv1 = nn.Conv2d(9, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-3, momentum=.1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-3, momentum=.1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-3, momentum=.1)
        self.pool = nn.MaxPool2d(2)
        self.drop1, self.drop2, self.dropg, self.dropout = nn.Dropout(.2), nn.Dropout(.25), nn.Dropout(.3), nn.Dropout(.4)
        self.surface = nn.Linear(128, 64)
        self.gravity1, self.gravity2 = nn.Linear(1, 16), nn.Linear(16, 16)
        self.fusion, self.output = nn.Linear(80, 64), nn.Linear(64, 1)
        self.register_buffer('gravity_mean', torch.tensor(float(gravity_mean)))
        self.register_buffer('gravity_variance', torch.tensor(max(float(gravity_variance), 1e-7)))

    def forward(self, image, gravity):
        x = self.drop1(self.pool(torch.relu(self.bn1(self.conv1(image)))))
        x = self.drop2(self.pool(torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.surface(x.mean((2, 3))))
        if self.gravity_dropout_after_norm:
            g = self.dropg((gravity - self.gravity_mean) / torch.sqrt(self.gravity_variance))
        else:
            g = (self.dropg(gravity) - self.gravity_mean) / torch.sqrt(self.gravity_variance)
        g = torch.relu(self.gravity1(g))
        g = torch.relu(self.gravity2(g))
        return self.output(self.dropout(torch.relu(self.fusion(torch.cat((x, g), dim=1))))).squeeze(1)


@lru_cache(maxsize=1)
def model_bundle():
    national_report = ARTIFACTS / 'national_fusion_report.json'
    if national_report.exists():
        report = json.loads(national_report.read_text(encoding='utf-8'))
        if report.get('eligible_for_deployment') is True:
            from backend.models.geo_intelligence.national_fusion import Ensemble
            checkpoint = ARTIFACTS / 'national_fusion.pt'
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report['model_sha256']:
                raise RuntimeError('National fusion model checksum mismatch')
            model = Ensemble(torch.load(checkpoint, map_location='cpu', weights_only=True)).eval()
            report.update(evaluation=report['cnn_oof'], evaluation_method=report['evaluation_scope'], calibrated=False,
                          score_meaning='Occurrence versus unknown-background screening index; not a reserve or occurrence probability.')
            return model, report
    candidate = ARTIFACTS / 'world_fusion_india_candidate.pt'
    candidate_metadata = ARTIFACTS / 'world_fusion_india_metadata.json'
    if candidate.exists() and candidate_metadata.exists():
        path, metadata_path = candidate, candidate_metadata
    else:
        path = ARTIFACTS / 'world_fusion.pt'
        metadata_path = ARTIFACTS / 'world_fusion_metadata.json'
    if not path.exists() or not metadata_path.exists():
        raise RuntimeError('World fusion model is not installed')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('model_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise RuntimeError('World fusion model checksum mismatch')
    model = WorldFusionCNN(metadata['gravity_mean'], metadata['gravity_variance'], **metadata.get('architecture_options', {}))
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model, metadata


@lru_cache(maxsize=1)
def gravity_grid():
    from scipy.io import netcdf_file
    from scipy.interpolate import RegularGridInterpolator
    path = ARTIFACTS / 'world_bouguer.grd'
    expected=json.loads((ARTIFACTS/'world_fusion_metadata.json').read_text(encoding='utf-8')).get('gravity_sha256')
    with path.open('rb') as stream:
        if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise RuntimeError('Gravity grid is a Git LFS pointer. Run python scripts/prepare_deployment.py during build.')
    if expected and file_sha256(path)!=expected:
        raise RuntimeError('World gravity grid checksum mismatch')
    # Keep the file open to allow mmap, avoiding a 222MB+ copy into RAM
    f = netcdf_file(path, 'r', mmap=True)
    lons = f.variables['x'][:]
    lats = f.variables['y'][:]
    z = f.variables['z'][:]
    if z.shape != (len(lats), len(lons)):
        raise ValueError('Global gravity grid is incomplete')
        
    class DirectInterpolator:
        def __init__(self, lats, lons, z):
            self.file = f
            self.lats, self.lons, self.z = lats, lons, z
            self.lat_step = (lats[-1] - lats[0]) / (len(lats) - 1)
            self.lon_step = (lons[-1] - lons[0]) / (len(lons) - 1)
        def __call__(self, pts):
            res = []
            for lat, lon in pts:
                i = int(round((lat - self.lats[0]) / self.lat_step))
                j = int(round((lon - self.lons[0]) / self.lon_step))
                i = max(0, min(i, len(self.lats)-1))
                j = max(0, min(j, len(self.lons)-1))
                res.append(float(self.z[i, j]))
            return res
            
    return DirectInterpolator(lats, lons, z)


def prepare_image(bands):
    from scipy import ndimage
    six = np.asarray(bands, dtype=np.float32)
    if six.ndim != 3 or six.shape[0] < 6:
        raise ValueError('Six reflectance bands are required')
    # Training and live cloud patches use a 64x64 central footprint. Local
    # supplied TIFFs are larger AOIs, so preserve that same ground scale.
    height, width = six.shape[1:]
    if height > 64 or width > 64:
        y0, x0 = max(0, (height-64)//2), max(0, (width-64)//2)
        six = six[:, y0:y0+64, x0:x0+64]
    six = six[:6].transpose(1, 2, 0) / 10000.0
    valid = np.isfinite(six).all(axis=-1) & (six > 0).all(axis=-1)
    yy, xx = np.where(valid)
    if len(yy) < 64:
        raise ValueError('Insufficient valid satellite pixels')
    six = np.clip(np.nan_to_num(six, nan=0, posinf=0, neginf=0), 0, 1)
    y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
    six = six[y0:y1, x0:x1].copy()
    mask = valid[y0:y1, x0:x1]
    if not mask.all():
        nearest = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        six = six[nearest[0], nearest[1]]
    six = ndimage.zoom(six, (64/six.shape[0], 64/six.shape[1], 1), order=1, prefilter=False).astype('float32')
    def index(i, j):
        den = six[..., i] + six[..., j]
        return np.divide(six[..., i]-six[..., j], den, out=np.zeros_like(den), where=np.abs(den)>1e-8)
    image = np.concatenate((six, np.stack((index(3, 2), index(1, 3), index(3, 4)), axis=-1)), axis=-1)
    return np.nan_to_num(image, nan=0, posinf=0, neginf=0).transpose(2, 0, 1)


def predict(image, latitude, longitude):
    model, metadata = model_bundle()
    gravity = float(gravity_grid()([[latitude, longitude]])[0])
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)[None])
    with torch.inference_mode():
        logit = float(model(tensor, torch.tensor([[gravity]], dtype=torch.float32))[0])
    temperature = float(metadata.get('temperature', 1.0))
    score = float(1/(1+np.exp(-np.clip(logit/temperature,-30,30))))
    return dict(latitude=float(latitude), longitude=float(longitude), score=score, gravity_mgal=gravity,
                status=('Regional experimental score · national validation unproven' if metadata.get('evaluation_independent') is False
                        else 'Experimental screening index · not a deposit confirmation'), model_version=metadata['version'],
                calibrated=bool(metadata.get('calibrated',False)),
                score_meaning=metadata.get('score_meaning', 'Uncalibrated screening score; not a field-validated occurrence probability.'),
                evaluation=metadata.get('evaluation'), evaluation_method=metadata.get('evaluation_method'))


def inspect_point(longitude, latitude):
    import rasterio
    from backend.cloud import fetch_patch
    from backend.exploration import footprint
    from backend.models.geo_intelligence.fusion_inputs import VERSION, read as read_aligned
    national = model_bundle()[1].get('preprocessing_version') == VERSION
    if national:
        from backend.national_imagery import fetch_patch
    try:
        path, source = (fetch_patch(longitude, latitude, size=64) if national
                        else fetch_patch(longitude, latitude, size=64, min_valid=.85, resolution_m=10))
        image_latitude, image_longitude = latitude, longitude
    except Exception as cloud_error:
        from backend.config import RASTERS
        from backend.store import distance
        manifest_path = ARTIFACTS / 'raster_manifest.json'
        candidates = []
        manifests = ((ARTIFACTS/'national_training'/'manifest.json', 0),) if national else ((ARTIFACTS / 'india_fusion_manifest.json', 0), (manifest_path, 1))
        for manifest, priority in manifests:
            if manifest.exists():
                for row in json.loads(manifest.read_text(encoding='utf-8')):
                    gap = distance(dict(latitude=latitude, longitude=longitude),
                                   dict(latitude=row.get('latitude', 0), longitude=row.get('longitude', 0)))
                    if gap <= 3:
                        candidates.append((priority, gap, row))
        candidates.sort(key=lambda item: (item[0], item[2].get('split') != 'test', item[1]))
        local = None
        for _, gap, row in candidates:
            original = Path(row.get('path', ''))
            path = original if original.exists() else ((ARTIFACTS/'national_imagery'/original.name) if national else RASTERS / original.parent.name / original.name)
            if not path.exists():
                continue
            try:
                with rasterio.open(path) as src:
                    candidate_image = read_aligned(path) if national else prepare_image(src.read())
                    candidate_crs = str(src.crs)
                local = (path, row, gap, candidate_image, candidate_crs)
                break
            except (ValueError, rasterio.errors.RasterioError):
                continue
        if local is None:
            message = ('Windows blocked the satellite connection (WinError 10013); no supplied 9-band TIFF is '
                       'available within 3 km. No point score was generated.'
                       if '10013' in str(cloud_error) or 'forbidden by its access permissions' in str(cloud_error).lower()
                       else f'Satellite imagery is unavailable and no supplied 9-band TIFF is within 3 km. No point score was generated. {str(cloud_error)[:180]}')
            raise RuntimeError(message) from cloud_error
        path, row, gap, image, crs = local
        image_latitude, image_longitude = float(row['latitude']), float(row['longitude'])
        source = dict(source='Supplied local 9-band TIFF', sample_split=row.get('split') or 'india_adaptation_training',
                      label_status=row.get('label_status', 'supplied_training_label'),
                      sample_label=int(row['label']) if row.get('label') is not None else None,
                      image_offset_km=round(float(gap), 3),
                      warning=('Score is for the nearby supplied image, not the clicked coordinate. '
                               'The TIFF is from the model dataset; this is not independent field evidence.'))
        result = predict(image, image_latitude, image_longitude)
        result['input_band_order'] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'NDVI', 'NDWI', 'NDMI']
        result['input_band_medians'] = [float(v) for v in np.median(image, axis=(1, 2))]
        result['input_valid_fraction'] = row.get('imagery', {}).get('valid_fraction', row.get('valid_fraction'))
        result.update(source=source, footprint=footprint(image_longitude, image_latitude, crs, 1280 if national else 640),
                      patch_width_m=1280 if national else 640,
                      requested_location=dict(latitude=float(latitude), longitude=float(longitude)),
                      status=f'Nearby supplied TIFF · {gap:.2f} km · training sample, not independent evidence',
                      score_status='local sample screening index; not a probability',
                      score_label='Screening score (0–100)')
        return result
    with rasterio.open(path) as src:
        image = read_aligned(path) if national else prepare_image(src.read())
        crs = str(src.crs)
    result = predict(image, latitude, longitude)
    result['input_band_order'] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'NDVI', 'NDWI', 'NDMI']
    result['input_band_medians'] = [float(v) for v in np.median(image, axis=(1, 2))]
    result['input_valid_fraction'] = source.get('valid_fraction')
    result.update(source=source, footprint=footprint(longitude, latitude, crs, 1280 if national else 640),
                  patch_width_m=1280 if national else 640,
                  score_status='sample-scaled screening index; not a probability',
                  score_label='Screening score (0–100)')
    return result


def scan_nearby(longitude, latitude):
    import math
    import rasterio
    from rasterio.warp import transform
    from backend.cloud import fetch_patch
    from backend.store import distance
    from backend.exploration import footprint
    from backend.models.geo_intelligence.fusion_inputs import VERSION, prepare as prepare_aligned
    national = model_bundle()[1].get('preprocessing_version') == VERSION
    if national:
        from backend.national_imagery import fetch_patch
    path, source = (fetch_patch(longitude, latitude, size=576) if national
                    else fetch_patch(longitude, latitude, size=1152, min_valid=.3, resolution_m=10))
    positions, arrays, missing = [], [], []
    with rasterio.open(path) as src:
        bands, crs, tf = src.read(), str(src.crs), src.transform
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if math.hypot(dx*1280, dy*1280) > 5000:
                    continue
                center, stride = (288, 64) if national else (576, 128)
                row, col = center+dy*stride, center+dx*stride
                x, y = tf*(col, row)
                lons, lats = transform(crs, 'EPSG:4326', [x], [y])
                lon, lat = float(lons[0]), float(lats[0])
                gap = distance(dict(longitude=longitude, latitude=latitude), dict(longitude=lon, latitude=lat))
                if gap > 5:
                    continue
                patch = bands[:, row-32:row+32, col-32:col+32]
                valid = np.isfinite(patch[:6]).all(axis=0) & (patch[:6] > 0).all(axis=0)
                entry = dict(longitude=lon, latitude=lat, distance_km=round(gap, 3), valid_fraction=float(valid.mean()))
                if valid.mean() < .8:
                    missing.append(entry)
                    continue
                try:
                    model_input = prepare_aligned(patch) if national else prepare_image(patch)
                    positions.append(entry)
                    arrays.append(model_input)
                except ValueError:
                    missing.append(entry)
        model, metadata = model_bundle()
        if arrays:
            grav = [float(gravity_grid()([[p['latitude'], p['longitude']]])[0]) for p in positions]
            with torch.inference_mode():
                logits = np.array([model(torch.from_numpy(arr[None]), torch.tensor([[g]], dtype=torch.float32)).numpy()[0] for arr, g in zip(arrays, grav)])
                scores = 1/(1+np.exp(-np.clip(logits/float(metadata.get('temperature',1.0)),-30,30)))
        else:
            scores = []
    features = []
    for p, score, grav in zip(positions, scores, grav if positions else []):
        features.append(dict(type='Feature', geometry=footprint(p['longitude'], p['latitude'], crs, 1280 if national else 640), properties={**p,
            'score':float(score), 'gravity_mgal':grav, 'name':'Nearby grid patch', 'date':source.get('acquisition_date'),
            'source':source.get('source'), 'model_version':metadata['version'], 'calibrated':False, 'high_score':bool(score>=.9)}))
    features.sort(key=lambda f: (-f['properties']['score'], f['properties']['distance_km']))
    hits=sum(f['properties']['high_score'] for f in features)
    message=f'{hits} sampled patches scored 90+ within 5 km.' if hits else 'No sampled patches scored 90+ within 5 km.'
    if missing:
        message+=' Scan incomplete: some patches lacked clear imagery.'
    message+=' Uncalibrated screening scores are not confirmed deposits or reserve estimates.'
    return dict(type='FeatureCollection', features=features, missing=missing, source=source, radius_km=5,
                grid_spacing_m=1280, total=len(features)+len(missing), scored=len(features), high_score_count=hits,
                incomplete=bool(missing), message=message)
