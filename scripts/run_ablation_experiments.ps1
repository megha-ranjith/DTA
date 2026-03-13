param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$configs = @(
    "configs/ablation_no_cross_attention.yaml",
    "configs/ablation_no_contrastive.yaml",
    "configs/ablation_no_physchem.yaml",
    "configs/ablation_no_kg.yaml"
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
Write-Output "All ablation experiments completed."
& $PythonExe scripts/compare_final_runs.py
