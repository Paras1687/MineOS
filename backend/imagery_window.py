import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds


def read_window(src, target, crs, size, categorical=False):
    bounds = rasterio.transform.array_bounds(size, size, target)
    window = from_bounds(*transform_bounds(crs, src.crs, *bounds), src.transform)
    window = Window(np.floor(window.col_off)-2, np.floor(window.row_off)-2,
                    np.ceil(window.width)+4, np.ceil(window.height)+4)
    window = window.intersection(Window(0, 0, src.width, src.height))
    raw = src.read(1, window=window).astype('float32')
    out = np.full((size, size), np.nan, dtype='float32')
    reproject(raw, out, src_transform=src.window_transform(window), src_crs=src.crs,
              src_nodata=0, dst_transform=target, dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.nearest if categorical else Resampling.bilinear,
              warp_mem_limit=16)
    return out
