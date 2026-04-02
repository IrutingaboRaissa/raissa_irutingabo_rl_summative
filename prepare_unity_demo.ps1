# Stages NutriVision replay JSON for Unity and optionally copies it into a Unity project.
# Requires: Python 3.13 (use standard build: py -3.13), dependencies from requirements.txt.
param(
    [string]$UnityProjectPath = "",
    [int]$Episodes = 3,
    [switch]$Trained
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$JsonOut = Join-Path $Root "outputs\unity\replay_trajectories.json"
$Args = @("unity_export.py", "--episodes", "$Episodes", "--out", $JsonOut)
if ($Trained) {
    $Args += "--algorithm", "ppo"
} else {
    $Args += "--random-policy"
}

Write-Host "Running: py -3.13 $($Args -join ' ')"
& py -3.13 @Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($UnityProjectPath) {
    $DestDir = Join-Path $UnityProjectPath "Assets\StreamingAssets"
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Copy-Item -Force $JsonOut (Join-Path $DestDir "replay_trajectories.json")
    Write-Host "[OK] Copied replay JSON to $DestDir"
} else {
    Write-Host ""
    Write-Host "Next: create a 3D (URP) Unity project, add TextMeshPro, then:"
    Write-Host "  Copy unity_bridge\NutriVisionReplayController.cs and NutriVisionUnityEnv.cs -> Assets\Scripts\"
    Write-Host "  Copy outputs\unity\replay_trajectories.json -> Assets\StreamingAssets\"
    Write-Host "  Attach NutriVisionReplayController to a GameObject, press Play."
}
