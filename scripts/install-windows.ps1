# Установка узла или координатора fastagg на Windows.
# Запуск из каталога репозитория, из PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\install-windows.ps1 -Role Node
#   .\scripts\install-windows.ps1 -Role Coordinator
#
# Скрипт создаёт venv, ставит пакет, открывает порты частной сети
# и регистрирует задачу планировщика на вход пользователя.
# Отдельной службы Windows без NSSM/WinSW здесь нет: задача при входе
# в систему — рабочий автозапуск. Трей не рисуется.

param(
    [ValidateSet("Node", "Coordinator")]
    [string]$Role = "Node",
    [string]$Python = "py",
    [switch]$NoAutostart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& $Python -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    & $Python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .

$Rules = @(
    @{ Name = "fastagg-discovery"; Protocol = "UDP"; Port = 45221 },
    @{ Name = "fastagg-node"; Protocol = "TCP"; Port = 47000 },
    @{ Name = "fastagg-rpc"; Protocol = "TCP"; Port = 50052 },
    @{ Name = "fastagg-dashboard"; Protocol = "TCP"; Port = "47080-47099" }
)
foreach ($Rule in $Rules) {
    $existing = Get-NetFirewallRule -DisplayName $Rule.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Remove-NetFirewallRule -DisplayName $Rule.Name
    }
    New-NetFirewallRule -DisplayName $Rule.Name -Direction Inbound -Protocol $Rule.Protocol -LocalPort $Rule.Port -Action Allow -Profile Private | Out-Null
    Write-Host "Правило брандмауэра: $($Rule.Name) порт $($Rule.Port)"
}

$Exe = Join-Path $Root ".venv\Scripts\fastagg.exe"
if ($Role -eq "Coordinator") {
    $Arguments = "coordinator"
} else {
    $Arguments = "node"
}

if (-not $NoAutostart) {
    $Action = New-ScheduledTaskAction -Execute $Exe -Argument $Arguments -WorkingDirectory $Root
    $Trigger = New-ScheduledTaskTrigger -AtLogOn
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName "fastagg-$Role" -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
    Write-Host "Автозапуск: задача fastagg-$Role при входе в систему."
}

Write-Host ""
Write-Host "Дальше:"
if ($Role -eq "Coordinator") {
    Write-Host "  .\.venv\Scripts\fastagg.exe coordinator"
    Write-Host "  Код сопряжения появится в консоли. Дашборд по умолчанию https://127.0.0.1:47080; если порт занят, адрес будет в этой же консоли."
} else {
    Write-Host "  .\.venv\Scripts\fastagg.exe join --url <ссылка fastagg:// с координатора>"
    Write-Host "  .\.venv\Scripts\fastagg.exe node"
}
Write-Host "RPC llama.cpp между машинами не шифруется. Не выставляйте порты в интернет."
