from datetime import datetime,timezone
import time,httpx
_cache={}

def forecast(lat,lon):
    key=(round(lat,3),round(lon,3)); previous=_cache.get(key)
    if previous and time.time()-previous['cached_at']<900:return previous
    try:
        r=httpx.get('https://api.open-meteo.com/v1/forecast',params=dict(latitude=lat,longitude=lon,
            current='temperature_2m,relative_humidity_2m,precipitation',daily='precipitation_sum,temperature_2m_mean',hourly='relative_humidity_2m',forecast_days=16,timezone='auto'),timeout=12)
        r.raise_for_status(); data=r.json(); days=[]
        for i,day in enumerate(data['daily']['time']):
            humidity=data['hourly']['relative_humidity_2m'][i*24:(i+1)*24]
            humidity=[v for v in humidity if v is not None]
            rain=data['daily']['precipitation_sum'][i]; temp=data['daily']['temperature_2m_mean'][i]
            if rain is None or temp is None or not humidity:continue
            days.append(dict(date=day,rainfall_mm=rain,temperature_c=temp,humidity_pct=sum(humidity)/len(humidity)))
        value=dict(status='live forecast',source='Open-Meteo / CC BY 4.0',source_url='https://open-meteo.com/',
                   current=data.get('current'),retrieved_at=datetime.now(timezone.utc).isoformat(),cached_at=time.time(),days=days)
        _cache[key]=value;return value
    except Exception as e:
        if previous:return {**previous,'status':'stale forecast','error':str(e)}
        return dict(status='unavailable',days=[],source='Open-Meteo',error=str(e),retrieved_at=None)
