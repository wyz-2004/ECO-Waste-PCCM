param(
    [switch]$SkipCity,
    [switch]$SkipDefra,
    [switch]$SkipCountry,
    [switch]$SkipEvidence,
    [switch]$SkipTechnical
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:ECOWASTE_PROJECT_ROOT = $repoRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Invoke-Experiment {
    param(
        [string]$Name,
        [string]$ScriptName
    )

    Write-Host "== $Name =="
    & $python (Join-Path $PSScriptRoot $ScriptName)
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipCity) {
    Invoke-Experiment -Name "City accuracy-feasibility experiments" -ScriptName "run_longterm_experiment.py"
}

if (-not $SkipTechnical) {
    Invoke-Experiment -Name "Dual-head analysis" -ScriptName "run_dual_head_analysis.py"
    Invoke-Experiment -Name "Technical ablations" -ScriptName "run_technical_depth_experiments.py"
    Invoke-Experiment -Name "Cost-aware verification" -ScriptName "build_cost_aware_verification.py"
}

if (-not $SkipEvidence) {
    Invoke-Experiment -Name "Guarded residual-risk evidence" -ScriptName "run_evidence_enhancement.py"
}

if (-not $SkipDefra) {
    Invoke-Experiment -Name "Defra external validation" -ScriptName "run_defra_external_validation.py"
}

if (-not $SkipCountry) {
    Invoke-Experiment -Name "Country transfer and stress-boundary experiments" -ScriptName "run_country_generalization.py"
}

Write-Host "All requested ECO-Waste-PCCM algorithm experiments completed."

