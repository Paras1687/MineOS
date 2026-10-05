from backend.server import app

if __name__ == '__main__':
    import json
    import urllib.request
    import uvicorn
    for port in range(8000, 8011):
        url = f'http://127.0.0.1:{port}'
        try:
            with urllib.request.urlopen(f'{url}/api/v1/health', timeout=1) as response:
                health = json.loads(response.read().decode('utf-8'))
                if health.get('status') == 'ready':
                    print(f'MineOS is already running at {url}; reusing it.')
                    break
        except Exception:
            pass
        try:
            print(f'Starting MineOS at {url}. Press Ctrl+C to stop it.')
            uvicorn.run(app, host='127.0.0.1', port=port)
            break
        except OSError as exc:
            print(f'Could not bind port {port} ({exc}); trying the next port.')
    else:
        raise RuntimeError('Could not start MineOS on ports 8000–8010.')
