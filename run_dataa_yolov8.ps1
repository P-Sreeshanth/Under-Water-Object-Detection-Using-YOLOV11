param(
    [string]$RunName = "",
    [ValidateSet("fast", "accurate")]
    [string]$Profile = "fast",
    [ValidateSet("single", "auto")]
    [string]$ClassMode = "auto",
    [ValidateRange(1, 30)]
    [int]$NumClasses = 12,
    [switch]$Resume,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe3110 = Join-Path $RepoRoot ".venv310\Scripts\python.exe"
$PythonExeDefault = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $PythonExe3110) {
    $PythonExe = $PythonExe3110
} else {
    $PythonExe = $PythonExeDefault
}

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe. Create the virtual environment first."
}

$Args = @(
    "yolov8_dataa_pipeline.py",
    "--repo-root", $RepoRoot,
    "--data-root", (Join-Path $RepoRoot "dataa"),
    "--output-root", (Join-Path $RepoRoot "runs\dataa_yolov8"),
    "--profile", $Profile,
    "--class-mode", $ClassMode,
    "--num-classes", $NumClasses
)

if ($RunName -ne "") {
    $Args += @("--run-name", $RunName)
}
if ($Resume) {
    $Args += "--resume"
}
if ($SkipTrain) {
    $Args += "--skip-train"
}

Write-Host "Running YOLOv8 dataa pipeline with args:" -ForegroundColor Cyan
Write-Host ($Args -join " ") -ForegroundColor Gray

& $PythonExe @Args
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline exited with code $LASTEXITCODE"
}

Write-Host "Pipeline completed successfully." -ForegroundColor Green
