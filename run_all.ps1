param(
    [switch]$SkipTests,
    [switch]$TrainIfMissing,
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Join-Path $repoRoot "supply-chain-rl"
$pythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python was not found at '$pythonExe'."
}

if (-not (Test-Path $projectDir)) {
    throw "Project directory was not found at '$projectDir'."
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"

Push-Location $projectDir
try {
    $requiredModel = "supply_chain_agent_a2c.zip"

    if (-not (Test-Path $requiredModel)) {
        if ($TrainIfMissing) {
            Write-Host "No trained model found. Running train.py to generate it..."
            & $pythonExe "train.py"
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }
        else {
            Write-Host "No '$requiredModel' found. The UI can still start, but trained-agent mode may fall back to an older checkpoint or fail."
            Write-Host "Run '.\run_all.ps1 -TrainIfMissing' if you want to generate the model first."
        }
    }

    if (-not $SkipTests) {
        Write-Host "Running environment checks..."
        & $pythonExe "test_env.py"
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    Write-Host "Starting the Gradio app from '$projectDir'..."

    if ($Background) {
        $process = Start-Process -FilePath $pythonExe -ArgumentList "app.py" -WorkingDirectory $projectDir -PassThru
        Write-Host "App started in the background. PID: $($process.Id)"
        Write-Host "If Gradio uses the default port, open http://127.0.0.1:7860"
    }
    else {
        & $pythonExe "app.py"
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
