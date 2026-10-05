"""Public Sentinel-2 L2A STAC -> fixed-footprint COG window, with scene mask."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from functools import lru_cache
import hashlib
from time import monotonic,sleep
import httpx,numpy as np,rasterio
import planetary_computer
import fsspec
import aiohttp
from rasterio.io import FilePath
from rasterio.warp import reproject,Resampling,transform
from rasterio.transform import from_origin
from backend.config import ARTIFACTS

@lru_cache(maxsize=32)
def fetch_patch(lon,lat,*,size=64,min_valid=.8,resolution_m=20):
    if size not in (64,576,1152):raise ValueError("Unsupported imagery size")
    if resolution_m not in (10,20):raise ValueError('Unsupported imagery resolution')
    if not -80<=lat<=84:raise ValueError('Sentinel-2 UTM coverage supported between 80S and 84N')
    end=datetime.now(timezone.utc);start=end-timedelta(days=365)
    request={'collections':['sentinel-2-l2a'],'intersects':{'type':'Point','coordinates':[lon,lat]},
             'datetime':f'{start.date()}/{end.date()}','query':{'eo:cloud_cover':{'lt':80}},'limit':4,
             'sortby':[{'field':'datetime','direction':'desc'}]}
    def search(payload):
        last=None
        for attempt in range(3):
            try:
                response=httpx.post('https://planetarycomputer.microsoft.com/api/stac/v1/search',json=payload,timeout=20)
                if response.status_code==429 or response.status_code>=500:
                    last=httpx.HTTPStatusError(f'Satellite catalog returned {response.status_code}',request=response.request,response=response)
                    sleep(min(4,1+attempt));continue
                response.raise_for_status();return response.json().get('features',[])
            except (httpx.TimeoutException,httpx.NetworkError) as exc:
                last=exc;sleep(min(4,1+attempt))
        raise RuntimeError(f'Satellite catalog unavailable after retries: {last}')
    scenes=search(request)
    request['limit']=32
    request['sortby']=[{'field':'eo:cloud_cover','direction':'asc'}]
    scenes+=search(request)
    if not scenes:raise ValueError('No Sentinel-2 scenes found in the last year')
    # Keep recent options, then try the clearest scenes from other dates.
    recent=scenes[:4]
    ordered=recent+sorted(scenes,key=lambda s:s['properties'].get('eo:cloud_cover',100))
    scenes=[];seen=set()
    for scene in ordered:
        date=scene['properties'].get('datetime','')[:10]
        if date in seen:continue
        seen.add(date);scenes.append(scene)
        if len(scenes)>=16:break
    zone=min(60,int((lon+180)//6)+1);crs=f'EPSG:{32600+zone if lat>=0 else 32700+zone}'
    x,y=transform('EPSG:4326',crs,[lon],[lat]); tf=from_origin(x[0]-size*resolution_m/2,y[0]+size*resolution_m/2,resolution_m,resolution_m)
    errors=[];started=monotonic();checked=0;best=0.0
    for scene in scenes:
        if monotonic()-started>150:break
        checked+=1
        arrays=[]; props=scene['properties']; assets=scene['assets']
        try:
            for name in ['SCL','B02','B03','B04','B08','B11','B12']:
                if monotonic()-started>200:raise TimeoutError('Imagery search time budget reached; retry to inspect more dates')
                asset=assets[name]
                # The SDK caches SAS tokens per storage account/container. Calling
                # the sign REST endpoint for every band was triggering 429s.
                url=planetary_computer.sign_url(asset['href'])
                out=np.full((size,size),np.nan,dtype='float32')
                # Python's verified-TLS range reader avoids Windows GDAL/SChannel failures.
                # A block cache reads only the needed COG windows, not full satellite scenes.
                with fsspec.open(url,mode='rb',block_size=256*1024,client_kwargs={'timeout':aiohttp.ClientTimeout(total=30)}) as remote:
                    with FilePath(remote) as virtual, virtual.open() as src:
                        reproject(rasterio.band(src,1),out,dst_transform=tf,dst_crs=crs,dst_nodata=np.nan,
                            resampling=Resampling.nearest if name=='SCL' else Resampling.bilinear)
                if name=='SCL':
                    scl=out
                    fraction=float(np.isin(scl,[4,5,6]).mean());best=max(best,fraction)
                    if fraction<min_valid:raise ValueError(f'Only {fraction:.0%} clear pixels')
                    continue
                if name!='SCL':
                    rb=asset.get('raster:bands',[{}])[0]
                    # Respect per-asset scaling. Otherwise require a known processing baseline.
                    if 'scale' in rb:
                        out=(out*float(rb['scale'])+float(rb.get('offset',0)))*10000
                    elif props.get('s2:processing_baseline') is not None:
                        if float(props['s2:processing_baseline'])>=4:out=out-1000
                    else:raise ValueError('Scene reflectance scaling unspecified')
                arrays.append(out)
            valid=np.isin(scl,[4,5,6]) & np.all(np.isfinite(arrays),axis=0)
            if valid.mean()<min_valid:raise ValueError('Too much local cloud, shadow, snow or missing imagery')
            six=np.stack(arrays);indices=[]
            for a,b in [(3,2),(1,3),(3,4)]:indices.append((six[a]-six[b])/np.maximum(six[a]+six[b],1e-6))
            stack=np.concatenate([six,np.stack(indices)]);stack[:,~valid]=np.nan
            cache=ARTIFACTS/'cloud';cache.mkdir(exist_ok=True)
            ident=hashlib.sha1(f'{lon:.6f}/{lat:.6f}/{scene["id"]}/{size}/{resolution_m}'.encode()).hexdigest()[:16];path=cache/f'{ident}.tif'
            with rasterio.open(path,'w',driver='GTiff',height=size,width=size,count=9,dtype='float32',crs=crs,transform=tf,nodata=np.nan) as dst:dst.write(stack)
            return path,dict(scene_id=scene['id'],acquisition_date=props.get('datetime'),cloud_pct=props.get('eo:cloud_cover'),
                source='Copernicus Sentinel-2 L2A / Microsoft Planetary Computer',valid_fraction=float(valid.mean()),
                resolution_m=resolution_m,footprint_m=size*resolution_m,
                warning='Cloud inference transfer is experimental; source training export harmonisation is not fully documented')
        except Exception as e:
            errors.append(type(e).__name__+': '+str(e).split('?')[0][:150])
            if '10013' in str(e):raise
    detail=next((e for e in reversed(errors) if not e.startswith('ValueError: Only')), '')
    raise RuntimeError(f'Checked {checked} satellite dates from the last year; no usable patch (at least {min_valid:.0%} clear pixels required; best mask {best:.0%}). '+detail)

