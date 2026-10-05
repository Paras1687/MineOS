import tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from backend import cloud

class CloudSearchTests(unittest.TestCase):
    def test_skips_cloudy_dates_before_loading_spectral_bands(self):
        scenes=[{'id':str(i),'properties':{'datetime':f'2026-09-{20-i:02d}T00:00:00Z','eo:cloud_cover':i,'s2:processing_baseline':4},'assets':{n:{'href':n} for n in ['SCL','B02','B03','B04','B08','B11','B12']}} for i in range(5)]
        response=MagicMock();response.status_code=200;response.json.return_value={'features':scenes}
        calls=[]
        def project(src,out,**kw):
            calls.append(kw['resampling'])
            out[:]=9 if len(calls)<=3 else (4 if len(calls)==4 else 3000)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cloud,'ARTIFACTS',Path(tmp)), patch.object(cloud.httpx,'post',return_value=response), patch.object(cloud.planetary_computer,'sign_url',side_effect=lambda v:v), patch.object(cloud.fsspec,'open'), patch.object(cloud,'FilePath'), patch.object(cloud.rasterio,'band'), patch.object(cloud,'reproject',side_effect=project):
            path,meta=cloud.fetch_patch(80,22)
            self.assertTrue(path.exists())
            self.assertEqual(meta['scene_id'],'3')
            self.assertEqual(meta['valid_fraction'],1)
            self.assertEqual(len(calls),10) # Four masks, only one six-band download.

if __name__=='__main__':unittest.main()
