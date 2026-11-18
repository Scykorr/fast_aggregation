# Поднимает координатор fastagg, ждёт модель и запускает RAG-5060 в WSL.
# Один профиль compose: core, если кластер готов, иначе только ollama.
param(
    [string]$RagPath = "D:\RAG 5060",
    [string]$Distro = "Ubuntu-24.04",
    [int]$TimeoutSec = 180,
    [string]$RagHealth = "http://127.0.0.1:28080/api/health"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "fastagg.exe"
if (-not (Test-Path $exe)) {
    $exe = "fastagg"
}

function To-WslPath([string]$Path) {
    if ($Path -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = ($Matches[2] -replace '\\', '/')
        return "/mnt/$drive/$rest"
    }
    return ($Path -replace '\\', '/')
}

Write-Host "Старт координатора"
Start-Process -FilePath $exe -ArgumentList "coordinator" -WindowStyle Hidden
& $exe wait-ready --timeout $TimeoutSec
$ready = ($LASTEXITCODE -eq 0)
$profileName = "core"
if (-not $ready) {
    Write-Host "Модель кластера не поднялась. RAG запустится с локальным Ollama."
    $profileName = "ollama"
}

if (-not (Test-Path $RagPath)) {
    Write-Error "Папка RAG не найдена: $RagPath"
    exit 1
}
$wslPath = To-WslPath $RagPath
$compose = "cd `"$wslPath`" && docker compose -f docker-compose.yml -f docker-compose.fastagg.yml --profile $profileName up -d"
Write-Host "docker compose --profile $profileName"
wsl -d $Distro -- bash -lc $compose
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose завершился с кодом $LASTEXITCODE"
    exit $LASTEXITCODE
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $RagHealth -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
            Write-Host "RAG отвечает: $RagHealth"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}
Write-Error "RAG не ответил: $RagHealth"
exit 1
