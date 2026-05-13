param(
    [string]$RepoUrl = "https://github.com/wenet-e2e/wekws.git",
    [string]$TargetDir = ".\third_party\wekws",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path (Get-Location) $Path)
}

$target = Resolve-ProjectPath $TargetDir

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git was not found in PATH."
}

if (-not (Test-Path -LiteralPath $target)) {
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    git clone --branch $Branch $RepoUrl $target
}
else {
    if (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
        throw "$target exists but is not a git repository."
    }

    git -C $target fetch origin
    git -C $target checkout $Branch
    git -C $target pull --ff-only origin $Branch
}

Write-Host "WeKWS source is ready:"
git -C $target remote -v
Write-Host "Branch:"
git -C $target branch --show-current
Write-Host "Commit:"
git -C $target rev-parse --short HEAD
