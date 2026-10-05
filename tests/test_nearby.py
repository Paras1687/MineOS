import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch,rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform
from backend.exploration import scan_nearby
class Constant(torch.nn.Module):
 def forward(self,x):return torch.full((len(x),1),3.)
class NearbyTests(unittest.TestCase):
 def test_radius_quality_and_calibration(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'area.tif';x,y=transform('EPSG:4326','EPSG:32644',[80],[22]);data=np.full((9,576,576),2000,dtype='float32');data[6:]=0;data[:,256:320,256:320]=np.nan
   with rasterio.open(p,'w',driver='GTiff',width=576,height=576,count=9,dtype='float32',crs='EPSG:32644',transform=from_origin(x[0]-5760,y[0]+5760,20,20)) as dst:dst.write(data)
   with patch('backend.cloud.fetch_patch',return_value=(p,{'acquisition_date':'2026-03-05','source':'test'})),patch('backend.exploration.model_bundle',return_value=(Constant(),{'calibration_slope':1,'calibration_intercept':-3})):
    r=scan_nearby(80,22)
   self.assertEqual(r['total'],45);self.assertEqual(r['scored'],44);self.assertTrue(r['incomplete']);self.assertEqual(r['high_score_count'],0)
   self.assertTrue(all(f['properties']['distance_km']<=5 for f in r['features']))
   self.assertTrue(all(f['properties']['score']==.5 for f in r['features']))
if __name__=='__main__':unittest.main()
