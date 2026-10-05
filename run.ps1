param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mineosPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $mineosPython)) {
    $mineosPython = Join-Path $PSScriptRoot '../mineos-env/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $mineosPython)) {
    $mineosPython = Join-Path $PSScriptRoot '../../work/mineos-env/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $mineosPython)) {
    throw 'Run setup.ps1 first, then run this launcher again.'
}
foreach ($candidatePort in @($Port) + (($Port + 1)..($Port + 10))) {
    $url = "http://127.0.0.1:$candidatePort"
    try {
        $health = Invoke-RestMethod -Uri "$url/api/v1/health" -TimeoutSec 2
        if ($health.status -eq 'ready') {
            Write-Host "MineOS is already running at $url. Reusing it; no second server was started."
            exit 0
        }
    } catch {
        # No healthy MineOS server responds on this port; try to start one.
    }
    Write-Host "Starting MineOS at $url. Keep this terminal open. Ctrl+C stops MineOS."
    & $mineosPython -m uvicorn backend.server:app --host 127.0.0.1 --port $candidatePort
    if ($LASTEXITCODE -eq 0) { exit 0 }
    Write-Warning "Port $candidatePort could not be used (possibly blocked or already reserved). Trying the next local port."
}
throw "Could not start MineOS on ports $Port-$($Port + 10). Close the conflicting local service or choose another port with: .\run.ps1 -Port 8100"
