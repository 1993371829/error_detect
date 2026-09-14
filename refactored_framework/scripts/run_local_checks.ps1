param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PreviousPythonPath = $env:PYTHONPATH
Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & $Python -B -m compileall -q src
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed" }
    & $Python -B -m pytest -q -o faulthandler_timeout=120
    if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
    & $Python -B -m hypergraph_ed doctor --config configs/local_smoke.yaml
    if ($LASTEXITCODE -ne 0) { throw "Local doctor failed" }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
