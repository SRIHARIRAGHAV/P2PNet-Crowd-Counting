$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "The Python launcher 'py' was not found. Install Python 3.13 first."
    }

    & py -3.13 -m venv .venv-cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.13 virtual environment."
    }

    $python = Join-Path $root ".venv-cpu\Scripts\python.exe"
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip."
    }
    & $python -m pip install -r requirements-cpu.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install CPU requirements."
    }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "The installed environment has dependency conflicts."
    }
}
finally {
    Pop-Location
}
