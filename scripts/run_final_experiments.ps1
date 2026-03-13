param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$configs = @(
    "configs/base_final.yaml",
    "configs/path1_final.yaml",
    "configs/path2_final.yaml",
    "configs/path3_final.yaml",
    "configs/path4_final.yaml"
)

foreach ($config in $configs) {
    Write-Output ""
    Write-Output "============================================================"
    Write-Output "Running $config"
    Write-Output "============================================================"
    & $PythonExe train_innovations.py --config $config
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $config"
    }
}

Write-Output ""
Write-Output "All final experiments completed."
& $PythonExe scripts/compare_final_runs.py
