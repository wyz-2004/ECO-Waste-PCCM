$ErrorActionPreference = "Stop"

$codeDir = $PSScriptRoot
$venvDir = Join-Path $codeDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $codeDir "requirements.txt"

if (-not (Test-Path -LiteralPath $requirements)) {
    throw "Missing requirements file: $requirements"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venvDir
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venvDir
    }
    else {
        throw "Python 3.11+ was not found. Install Python, then rerun this script."
    }
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $requirements

Write-Output "Environment ready: $venvPython"
