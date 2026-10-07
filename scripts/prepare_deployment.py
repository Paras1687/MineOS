import hashlib
import json
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    artifacts = ROOT/'backend'/'artifacts'
    path = artifacts/'world_bouguer.grd'
    expected = json.loads((artifacts/'world_fusion_metadata.json').read_text())['gravity_sha256']
    if path.exists() and checksum(path) == expected:
        print('Gravity grid verified')
        return
    temporary = path.with_suffix('.download')
    url = 'https://media.githubusercontent.com/media/Paras1687/MineOS/main/backend/artifacts/world_bouguer.grd'
    try:
        with httpx.stream('GET', url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_bytes(1024*1024):
                    stream.write(chunk)
        if checksum(temporary) != expected:
            raise RuntimeError('Downloaded gravity grid checksum mismatch')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print('Gravity grid installed and verified')


if __name__ == '__main__':
    main()
