"""Identical physical-footprint preparation for local and cloud rasters."""
import re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform

BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'NDVI', 'NDWI', 'NDMI']
SIZE = 64
FOOTPRINT_M = 1280

def site_key(name):
    name = re.sub(r'^Background_[\d.]+km_from_', '', name, flags=re.I)
    name = re.sub(r'_\d{4}-\d{2}-\d{2}_Stacked_9Band.*$', '', name)
    return re.sub(r'[^a-z0-9]', '', name.lower())

def read_patch(path, lon=None, lat=None):
    with rasterio.open(path) as src:
        if src.count != 9 or not src.crs or not src.crs.is_projected:
            raise ValueError('Expected a georeferenced, projected, nine-band raster')
        if src.crs.linear_units != 'metre':
            raise ValueError('Projected raster must use metres')
        if lon is None:
            x, y = (src.bounds.left+src.bounds.right)/2, (src.bounds.top+src.bounds.bottom)/2
            lons, lats = transform(src.crs, 'EPSG:4326', [x], [y])
            lon, lat = lons[0], lats[0]
        else:
            xs, ys = transform('EPSG:4326', src.crs, [lon], [lat]); x, y = xs[0], ys[0]
        half = FOOTPRINT_M/2
        window = from_bounds(x-half, y-half, x+half, y+half, src.transform)
        arr = src.read(window=window, out_shape=(9, SIZE, SIZE), boundless=True,
                       fill_value=float('nan'), resampling=Resampling.bilinear).astype('float32')
        valid = np.all(np.isfinite(arr), axis=0) & np.any(arr[:6] != 0, axis=0)
        fraction = float(valid.mean())
        if fraction < .8:
            raise ValueError(f'Only {fraction:.0%} valid pixels in 1.28 km footprint')
        arr[:, ~valid] = np.nan
        # Raster band descriptions are preferred; missing descriptions are explicitly tracked.
        descriptions = list(src.descriptions)
        checks = []
        for a,b,k in [(3,2,6),(1,3,7),(3,4,8)]:
            expected = (arr[a]-arr[b]) / np.maximum(arr[a]+arr[b], 1e-6)
            checks.append(float(np.nanmedian(np.abs(expected-arr[k]))))
        if max(checks) > .08:
            raise ValueError(f'Index bands differ from NDVI/NDWI/NDMI contract: {checks}')
        meta = dict(longitude=lon, latitude=lat, width=src.width, height=src.height,
                    crs=str(src.crs), valid_fraction=fraction, index_errors=checks,
                    band_descriptions=descriptions)
        return arr, meta

def normalize(arr):
    """Fixed Sentinel scaled-reflectance contract; zero-fill masked pixels after scaling."""
    x = arr.copy()
    x[..., :6, :, :] = np.clip(x[..., :6, :, :]/10000, 0, 1.5)
    x[..., 6:, :, :] = np.clip(x[..., 6:, :, :], -1, 1)
    return np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
