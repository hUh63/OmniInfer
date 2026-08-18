param(
    [string]$Installer = (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\install.ps1")
)

$ErrorActionPreference = "Stop"
$testRoot = Join-Path ([IO.Path]::GetTempPath()) "omniinfer-installer-test-$([Guid]::NewGuid().ToString('N'))"

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "FAIL: $Message" }
}

function New-FixtureRelease {
    param([string]$Version, [string]$ReleaseRoot, [string]$FixtureExecutable)

    $stage = Join-Path $testRoot "stage-$Version"
    $package = Join-Path $stage "OmniInfer"
    $releaseDir = Join-Path $ReleaseRoot $Version
    New-Item -ItemType Directory -Force -Path $package, $releaseDir | Out-Null
    Copy-Item -LiteralPath $FixtureExecutable -Destination (Join-Path $package "omniinfer.exe")
    "@echo off`r`n`"%~dp0omniinfer.exe`" %*`r`nexit /b %errorlevel%`r`n" |
        Set-Content -LiteralPath (Join-Path $package "omniinfer.cmd") -Encoding ASCII
    "& (Join-Path `$PSScriptRoot 'omniinfer.exe') @args`r`nexit `$LASTEXITCODE`r`n" |
        Set-Content -LiteralPath (Join-Path $package "omniinfer.ps1") -Encoding UTF8

    $assetName = "omniinfer-$Version-windows-x64.zip"
    $archive = Join-Path $releaseDir $assetName
    Compress-Archive -Path $package -DestinationPath $archive
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    "$sha  $assetName" | Set-Content -LiteralPath (Join-Path $releaseDir "checksums.txt") -Encoding ASCII
}

New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $releaseRoot = Join-Path $testRoot "releases\download"
    if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) {
        $fixtureExecutable = Join-Path $testRoot "fixture.exe"
        Add-Type -TypeDefinition @'
using System;

public static class InstallerFixture
{
    public static int Main(string[] args)
    {
        Console.WriteLine("omniinfer fixture");
        return 0;
    }
}
'@ -Language CSharp -OutputAssembly $fixtureExecutable -OutputType ConsoleApplication
    } else {
        $fixtureCommand = Get-Command git -ErrorAction Stop
        $fixtureExecutable = $fixtureCommand.Source
    }
    New-FixtureRelease v1.2.3 $releaseRoot $fixtureExecutable

    $installDir = Join-Path $testRoot "installed"
    & $Installer -Version v1.2.3 -BaseUrl $releaseRoot -Target windows-x64 -InstallDir $installDir -NoPathUpdate
    Assert-True (Test-Path -LiteralPath (Join-Path $installDir "omniinfer.exe")) "executable was not installed"
    Assert-True (Test-Path -LiteralPath (Join-Path $installDir "omniinfer.cmd")) "cmd launcher was not installed"
    Assert-True (Test-Path -LiteralPath (Join-Path $installDir "omniinfer.ps1")) "PowerShell launcher was not installed"

    # Reinstalling the same archive must remain safe.
    & $Installer -Version 1.2.3 -BaseUrl $releaseRoot -Target windows-x64 -InstallDir $installDir -NoPathUpdate
    $beforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $installDir "omniinfer.exe")).Hash

    New-FixtureRelease v1.2.4 $releaseRoot $fixtureExecutable
    "$(('0' * 64))  omniinfer-v1.2.4-windows-x64.zip" |
        Set-Content -LiteralPath (Join-Path $releaseRoot "v1.2.4\checksums.txt") -Encoding ASCII
    $checksumFailed = $false
    try {
        & $Installer -Version v1.2.4 -BaseUrl $releaseRoot -Target windows-x64 -InstallDir $installDir -NoPathUpdate
    } catch {
        $checksumFailed = $_.Exception.Message -like "*checksum mismatch*"
    }
    Assert-True $checksumFailed "checksum mismatch was accepted"
    $afterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $installDir "omniinfer.exe")).Hash
    Assert-True ($beforeHash -eq $afterHash) "failed install overwrote the existing executable"

    $invalidTargetFailed = $false
    try {
        & $Installer -Version v1.2.3 -BaseUrl $releaseRoot -Target macos-arm64 -InstallDir $installDir -NoPathUpdate
    } catch {
        $invalidTargetFailed = $_.Exception.Message -like "*unsupported target*"
    }
    Assert-True $invalidTargetFailed "unsupported target was accepted"
    Write-Host "release installer PowerShell tests passed"
} finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -Recurse -Force -LiteralPath $testRoot
    }
}
