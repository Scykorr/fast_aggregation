# Проверяет собранный fastagg.exe: scan и HTTP узла. Без сети и без трея.
param(
    [Parameter(Mandatory = $true)][string]$Exe
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Exe)) { throw "Нет файла $Exe" }
$env:FA_HOME = Join-Path $env:TEMP "fastagg-smoke"
New-Item -ItemType Directory -Force -Path $env:FA_HOME | Out-Null

& $Exe scan
if ($LASTEXITCODE -ne 0) { throw "scan завершился с кодом $LASTEXITCODE" }

$proc = Start-Process -FilePath $Exe -ArgumentList @(
    "node", "--open", "--insecure-http", "--bind", "127.0.0.1",
    "--http-port", "47099", "--discovery-port", "45999", "--name", "ci-node"
) -PassThru -WindowStyle Hidden
try {
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:47099/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200 -and $response.Content -match "ok") {
                $ok = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ok) { throw "Узел не открыл /health за 60 секунд" }
    Write-Host "smoke ok"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
