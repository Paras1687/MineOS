import hashlib, re
import pandas as pd
from functools import lru_cache
from backend.config import MASTER

def grade_range(text):
    text=str(text)
    # Keep manganese component only, never phosphorus or MnO2 as elemental Mn.
    if 'MnO2' in text or '+' in text or '>' in text: return None
    text=re.split(r',?\s*(?:and\s+)?P\s*\(',text)[0]
    nums=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',text)]
    if not nums or max(nums)>65 or min(nums)<0: return None
    return [min(nums),max(nums)]

@lru_cache(maxsize=1)
def catalog():
    d=pd.read_csv(MASTER).fillna('')
    for col in ['Latitude','Longitude']:
        d[col]=pd.to_numeric(d[col].astype(str).str.strip(),errors='coerce')
    d=d.dropna(subset=['Latitude','Longitude'])
    d=d[d.Latitude.between(-90,90)&d.Longitude.between(-180,180)]
    # CSV repeats five India sites with sub-millimetre coordinate differences.
    # Use the same coordinate precision for deduplication and stable identity.
    d=d.assign(_lat=d.Latitude.round(5),_lon=d.Longitude.round(5))
    d=d.drop_duplicates(['_lat','_lon','Target'])
    result=[]
    for _,r in d.iterrows():
        uid=hashlib.sha1(f"{r.Latitude:.5f},{r.Longitude:.5f}".encode()).hexdigest()[:10]
        result.append(dict(id='SITE-'+uid,name=r.Mine_Name,latitude=float(r.Latitude),longitude=float(r.Longitude),
            source=r.Source,target=int(r.Target),country='India' if 'India' in r.Source else 'International',
            geology=r.Geology,lithology=r.Lithology,host_rock=r.Host_Rock,formation=r.Formation,age=r.Age,
            drill_depth_reported=r.Drill_Depth,ore_thickness_reported=r.Ore_Thickness,
            reported_grade=r.Grade,grade_range=grade_range(r.Grade),reported_resource=r.Reserve,
            resource_unit='Unverified source tons / unspecified units; not converted to metric tonnes',
            operating_status='Unverified occurrence; mine operation not established by this CSV'))
    return result
