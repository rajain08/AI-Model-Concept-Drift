param(
    [string]$TaskName = "AI Model Concept Drift Data Pipeline",
    [string]$StartTime = "02:00"
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = (Get-Command python).Source
$script = Join-Path $repo "scripts\run_data_pipeline.py"
$log = Join-Path $repo "warehouse\pipeline.log"

if (-not (Test-Path $python)) {
    throw "Python was not found on PATH."
}

$action = "`"$python`" `"$script`" --log-file `"$log`""
schtasks.exe /Create /F /SC DAILY /TN $TaskName /TR $action /ST $StartTime | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Task Scheduler registration failed with exit code $LASTEXITCODE."
}

Write-Host "Registered '$TaskName' to run daily at $StartTime."
Write-Host "Verify with: schtasks.exe /Query /TN `"$TaskName`" /V /FO LIST"
