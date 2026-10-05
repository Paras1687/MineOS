"""Package only the active application, supplied training data and evaluated models."""
from pathlib import Path
import shutil,zipfile,hashlib,json

root=Path(__file__).resolve().parents[1]
out=root.parents[1]/'outputs'
target=out/'MineOS-AI'
out.mkdir(exist_ok=True)
files=[]
files += [root/name for name in ['README.md','MODEL_AUDIT.md','requirements.txt','setup.ps1','run.ps1','Start-MineOS.cmd','.gitignore']]
files += list((root/'backend').glob('*.py'))
files += [root/'backend/models/geo_intelligence'/name for name in ['raster.py','network.py','world_fusion.py','fusion_inputs.py','national_fusion.py']]
files += list((root/'backend/models/production_forecast').glob('*.csv'))
files += [root/'backend/data'/name for name in ['Manganese_Master_Dataset.csv','ai4i2020_train_FINAL.csv','ai4i2020_test_FINAL.csv']]
files += [p for p in (root/'backend/artifacts').iterdir() if p.is_file() and p.name!='patches.npy']
files += [p for p in (root/'backend/artifacts/national_training').glob('*.json')]
files += [p for p in (root/'backend/artifacts/national_training').glob('*.npy')]
files += [p for p in (root/'backend/artifacts/national_imagery').glob('*') if p.suffix in ('.json','.tif')]
files += list((root/'frontend').glob('*.html'))
files += [root/'frontend/css/explorer-console.css',root/'frontend/js/explorer-console.js',root/'frontend/css/portal.css',root/'frontend/css/exploration.css',root/'frontend/js/portal.js',root/'frontend/js/features.js',root/'frontend/js/estimation.js']
files += [p for p in (root/'frontend/vendor').rglob('*') if p.is_file()]
files += list((root/'scripts').glob('*.py'))+list((root/'tests').glob('*.py'))
manifest={}
for p in files:
    rel=p.relative_to(root);dest=target/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)
    manifest[rel.as_posix()]=hashlib.sha256(dest.read_bytes()).hexdigest()
(target/'SHA256.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(out/'MineOS-AI.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in manifest:z.write(target/rel,'MineOS-AI/'+rel)
    z.write(target/'SHA256.json','MineOS-AI/SHA256.json')
print(f'{len(manifest)} files; {(out/"MineOS-AI.zip").stat().st_size/1e6:.2f} MB -> {out}')
