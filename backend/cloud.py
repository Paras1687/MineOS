"""Public Sentinel-2 L2A STAC -> fixed-footprint COG window, with scene mask."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from functools import lru_cache
import hashlib
import json
from time import monotonic,sleep
import httpx,numpy as np,rasterio
import planetary_computer
import fsspec
import aiohttp
from rasterio.io import FilePath
from rasterio.warp import reproject,Resampling,transform
from rasterio.transform import from_origin
from backend.config import ARTIFACTS
from backend.imagery_window import read_window

def search_catalog(payload):
    last=None
    for attempt in range(4):
        try:
            response=httpx.post('https://planetarycomputer.microsoft.com/api/stac/v1/search',json=payload,timeout=20)
            response.raise_for_status()
            return response.json().get('features',[])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code!=429 and exc.response.status_code<500:raise
            last=exc
        except (httpx.TransportError,OSError) as exc:
            last=exc
        if attempt<3:sleep(2**attempt)
    raise RuntimeError(f'Satellite catalog unavailable after retries: {last}') from last

@lru_cache(maxsize=32)
def fetch_patch(lon,lat,*,size=64,min_valid=.8,resolution_m=20):
    if size not in (64,576,1152):raise ValueError("Unsupported imagery size")
    if resolution_m not in (10,20):raise ValueError('Unsupported imagery resolution')
    if not -80<=lat<=84:raise ValueError('Sentinel-2 UTM coverage supported between 80S and 84N')
    cache=ARTIFACTS/'cloud';cache.mkdir(exist_ok=True)
    key=hashlib.sha256(f'v1/{lon:.8f}/{lat:.8f}/{size}/{resolution_m}/{min_valid}'.encode()).hexdigest()
    sidecar=cache/f'{key}.json'
    if sidecar.exists():
        try:
            entry=json.loads(sidecar.read_text())
            saved=cache/Path(entry['file']).name
            if hashlib.sha256(saved.read_bytes()).hexdigest()==entry['sha256']:
                return saved,{**entry['metadata'],'cached':True}
        except (OSError,ValueError,KeyError):pass
    end=datetime.now(timezone.utc);start=end-timedelta(days=365)
    request={'collections':['sentinel-2-l2a'],'intersects':{'type':'Point','coordinates':[lon,lat]},
             'datetime':f'{start.date()}/{end.date()}','query':{'eo:cloud_cover':{'lt':80}},'limit':4,
             'sortby':[{'field':'datetime','direction':'desc'}]}
    scenes=search_catalog(request)
    request['limit']=32
    request['sortby']=[{'field':'eo:cloud_cover','direction':'asc'}]
    try:scenes+=search_catalog(request)
    except RuntimeError:
        if not scenes:raise
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
        stack=np.empty((9,size,size),dtype='float32')
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
                        out=read_window(src,tf,crs,size,categorical=name=='SCL')
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
                stack[len(arrays)]=out
                arrays.append(stack[len(arrays)])
            valid=np.isin(scl,[4,5,6]) & np.all(np.isfinite(arrays),axis=0)
            if valid.mean()<min_valid:raise ValueError('Too much local cloud, shadow, snow or missing imagery')
            six=stack[:6]
            for i,(a,b) in enumerate([(3,2),(1,3),(3,4)]):stack[6+i]=(six[a]-six[b])/np.maximum(six[a]+six[b],1e-6)
            stack[:,~valid]=np.nan
            cache=ARTIFACTS/'cloud';cache.mkdir(exist_ok=True)
            ident=hashlib.sha1(f'{lon:.6f}/{lat:.6f}/{scene["id"]}/{size}/{resolution_m}'.encode()).hexdigest()[:16];path=cache/f'{ident}.tif'
            with rasterio.open(path,'w',driver='GTiff',height=size,width=size,count=9,dtype='float32',crs=crs,transform=tf,nodata=np.nan) as dst:dst.write(stack)
            metadata=dict(scene_id=scene['id'],acquisition_date=props.get('datetime'),cloud_pct=props.get('eo:cloud_cover'),
                source='Copernicus Sentinel-2 L2A / Microsoft Planetary Computer',valid_fraction=float(valid.mean()),
                resolution_m=resolution_m,footprint_m=size*resolution_m,
                warning='Cloud inference transfer is experimental; source training export harmonisation is not fully documented')
            sidecar.write_text(json.dumps(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=metadata)))
            return path,metadata
        except Exception as e:
            errors.append(type(e).__name__+': '+str(e).split('?')[0][:150])
            if '10013' in str(e):raise
    detail=next((e for e in reversed(errors) if not e.startswith('ValueError: Only')), '')
    raise RuntimeError(f'Checked {checked} satellite dates from the last year; no usable patch (at least {min_valid:.0%} clear pixels required; best mask {best:.0%}). '+detail)
