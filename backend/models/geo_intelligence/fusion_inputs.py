import numpy as np
from scipy import ndimage

VERSION = 's2-aligned-20m-1280m-v2'
BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'NDVI', 'NDWI', 'NDMI']


def prepare(bands, min_valid=.85):
    a = np.asarray(bands, dtype=np.float32)
    if a.ndim != 3 or a.shape[0] < 6 or a.shape[1:] != (64, 64):
        raise ValueError('Aligned 64x64 bands on the 20 m grid are required')
    six = a[:6] / 10000
    valid = np.isfinite(six).all(0) & (six > 0).all(0) & (six <= 1.5).all(0)
    if valid.mean() < min_valid:
        raise ValueError(f'Only {valid.mean():.0%} usable pixels; {min_valid:.0%} required')
    if not valid.all():
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        six = six[:, nearest[0], nearest[1]]
    indices = [(six[i]-six[j]) / np.maximum(six[i]+six[j], 1e-8)
               for i, j in [(3, 2), (1, 3), (3, 4)]]
    return np.concatenate([six, np.stack(indices)]).astype('float32')


def read(path):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    with rasterio.open(path) as src:
        if not src.crs or not src.crs.is_projected or src.crs.linear_units != 'metre':
            raise ValueError('Georeferenced metre-based imagery required')
        if src.count != 9:
            raise ValueError('Nine bands required')
        x, y = src.transform * (src.width/2, src.height/2)
        window = from_bounds(x-640, y-640, x+640, y+640, src.transform)
        bands = src.read(window=window, out_shape=(9, 64, 64), boundless=True,
                         fill_value=np.nan, resampling=Resampling.bilinear)
    return prepare(bands)
