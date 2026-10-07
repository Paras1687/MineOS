import hashlib
import json
import threading
import time
from pathlib import Path

import aiohttp
import fsspec
import httpx
import numpy as np
import planetary_computer
import rasterio
from rasterio.io import FilePath
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform, transform_bounds
from rasterio.windows import from_bounds

from backend.config import ARTIFACTS
from backend.cloud import search_catalog
from backend.models.geo_intelligence.fusion_inputs import BANDS, VERSION

DATE_RANGE = '2025-01-01/2025-12-31'
_sign_lock = threading.Lock()


def fetch_patch(lon, lat, size=64):
    if not np.isfinite([lat, lon]).all() or not (-80 <= lat <= 84 and -180 <= lon <= 180):
        raise ValueError('Invalid Sentinel coordinates')
    if size not in (64, 576):
        raise ValueError('Unsupported patch size')
    cache = ARTIFACTS / 'national_imagery'
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(f'{lon:.6f}/{lat:.6f}/{size}/{DATE_RANGE}/{VERSION}'.encode()).hexdigest()[:24]
    path, sidecar = cache/f'{key}.tif', cache/f'{key}.json'
    if path.exists() and sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']:
            return path, meta
    body = dict(collections=['sentinel-2-l2a'], intersects=dict(type='Point', coordinates=[lon, lat]),
                datetime=DATE_RANGE, query={'eo:cloud_cover': {'lt': 50}}, limit=16,
                sortby=[{'field': 'eo:cloud_cover', 'direction': 'asc'}])
    scenes = search_catalog(body)
    crs = f'EPSG:{(32600 if lat >= 0 else 32700)+min(60, int((lon+180)//6)+1)}'
    x, y = transform('EPSG:4326', crs, [lon], [lat])
    target = from_origin(x[0]-size*10, y[0]+size*10, 20, 20)
    bounds = (x[0]-size*10, y[0]-size*10, x[0]+size*10, y[0]+size*10)
    errors = []
    for scene in scenes[:8]:
        try:
            bands = []
            for name in ['SCL']+BANDS[:6]:
                asset = scene['assets'][name]
                with _sign_lock:
                    url = planetary_computer.sign_url(asset['href'])
                with fsspec.open(url, mode='rb', block_size=256*1024,
                                client_kwargs={'timeout': aiohttp.ClientTimeout(total=25)}) as remote:
                    with FilePath(remote) as virtual, virtual.open() as src:
                        window = from_bounds(*transform_bounds(crs, src.crs, *bounds), src.transform)
                        window = rasterio.windows.Window(np.floor(window.col_off)-2, np.floor(window.row_off)-2,
                                                         np.ceil(window.width)+4, np.ceil(window.height)+4)
                        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                        raw = src.read(1, window=window).astype('float32')
                        arr = np.full((size, size), np.nan, dtype='float32')
                        reproject(raw, arr, src_transform=src.window_transform(window), src_crs=src.crs,
                                  dst_transform=target, dst_crs=crs, src_nodata=0, dst_nodata=np.nan,
                                  resampling=Resampling.nearest if name == 'SCL' else Resampling.bilinear)
                if name == 'SCL':
                    valid = np.isin(arr, [4, 5, 6])
                    if valid.mean() < .85:
                        raise ValueError(f'Clear pixel fraction {valid.mean():.2f}')
                    continue
                rb = asset.get('raster:bands', [{}])[0]
                scale = float(rb.get('scale', .0001))
                offset = rb.get('offset')
                if offset is None:
                    baseline = scene['properties'].get('s2:processing_baseline')
                    if baseline is None:
                        raise ValueError('Unknown reflectance offset')
                    offset = -.1 if float(baseline) >= 4 else 0
                bands.append((arr*scale+float(offset))*10000)
            six = np.stack(bands)
            valid &= np.isfinite(six).all(0) & (six > 0).all(0) & (six <= 15000).all(0)
            if valid.mean() < .85:
                raise ValueError('Insufficient valid reflectance')
            indices = [(six[i]-six[j])/np.maximum(six[i]+six[j], 1e-8) for i, j in [(3, 2), (1, 3), (3, 4)]]
            stack = np.concatenate([six, np.stack(indices)])
            stack[:, ~valid] = np.nan
            tmp = path.with_suffix('.tmp.tif')
            with rasterio.open(tmp, 'w', driver='GTiff', width=size, height=size, count=9,
                               crs=crs, transform=target, dtype='float32', nodata=np.nan) as dst:
                dst.write(stack)
                dst.descriptions = tuple(BANDS)
                dst.update_tags(preprocessing=VERSION)
            tmp.replace(path)
            meta = dict(latitude=lat, longitude=lon, scene_id=scene['id'], acquisition_date=scene['properties']['datetime'],
                        source='Copernicus Sentinel-2 L2A / Microsoft Planetary Computer', date_range=DATE_RANGE,
                        preprocessing=VERSION, resolution_m=20, footprint_m=size*20,
                        valid_fraction=float(valid.mean()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            sidecar.write_text(json.dumps(meta, indent=2))
            return path, meta
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {str(exc).split(chr(63))[0][:120]}')
    raise RuntimeError('No aligned clear imagery: '+('; '.join(errors[-2:]) or 'no scenes'))
