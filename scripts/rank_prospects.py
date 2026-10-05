import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from backend.config import ARTIFACTS
from backend.exploration import model_bundle
torch.set_num_threads(4)
d=json.loads((ARTIFACTS/'raster_manifest.json').read_text());x=np.load(ARTIFACTS/'patches.npy',mmap_mode='r');model,meta=model_bundle();rows=[]
with torch.no_grad():
    for i in range(0,len(d),64):
        logits=model(torch.tensor(np.array(x[i:i+64]))).numpy()
        probs=1/(1+np.exp(-np.clip(meta['calibration_slope']*logits+meta['calibration_intercept'],-30,30)))
        for r,p in zip(d[i:i+64],probs):
            rows.append(dict(name=Path(r['path']).stem,latitude=r['latitude'],longitude=r['longitude'],score=round(float(p),4),
                split=r['split'],known_label=r['label'],valid_fraction=r['valid_fraction'],date=r['date'],
                recommendation='Field survey / verify geology',reason='Ranked by model score; no mine extraction approval or reserve valuation implied'))
rows.sort(key=lambda r:r['score'],reverse=True)
(ARTIFACTS/'prospect_ranking.json').write_text(json.dumps(rows,indent=2));print('Ranked',len(rows),'patches')
