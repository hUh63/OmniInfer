# Lightweight OmniInfer CLI installer for Windows x64.
#
# Usage:
#   irm https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.ps1 | iex
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install.ps1 -Version v0.3.24

param(
    [string]$Version = $(if ($env:OMNIINFER_VERSION) { $env:OMNIINFER_VERSION } else { "latest" }),
    [string]$InstallDir = $(if ($env:OMNIINFER_INSTALL_DIR) {
        $env:OMNIINFER_INSTALL_DIR
    } else {
        Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "Programs\OmniInfer\bin"
    }),
    [string]$Repository = "omnimind-ai/OmniInfer",
    [string]$BaseUrl = "",
    [string]$ApiUrl = "https://api.github.com",
    [string]$Target = "",
    [switch]$NoPathUpdate,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" }
function Write-Ok { param([string]$Message) Write-Host "[ OK ] $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Warning $Message }
function Stop-Install { param([string]$Message) throw "OmniInfer installer: $Message" }

function Resolve-ReleaseVersion {
    param([string]$RequestedVersion)

    if ($RequestedVersion -eq "latest") {
        $uri = "$($ApiUrl.TrimEnd('/'))/repos/$Repository/releases/latest"
        try {
            $response = Invoke-RestMethod -Uri $uri -Headers @{ "User-Agent" = "OmniInfer-installer" }
        } catch {
            Stop-Install "failed to query latest release from ${uri}: $($_.Exception.Message)"
        }
        $RequestedVersion = [string]$response.tag_name
        if (-not $RequestedVersion) {
            Stop-Install "latest release response did not contain tag_name"
        }
    }

    if ($RequestedVersion -notmatch '^v?[0-9][0-9A-Za-z._-]*$') {
        Stop-Install "invalid release version: $RequestedVersion"
    }
    if ($RequestedVersion.StartsWith("v")) { return $RequestedVersion }
    return "v$RequestedVersion"
}

function Resolve-Target {
    if ($Target) {
        if ($Target -ne "windows-x64") { Stop-Install "unsupported target: $Target" }
        return $Target
    }

    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        Stop-Install "this installer supports Windows x64 only"
    }
    $architecture = if ($env:PROCESSOR_ARCHITEW6432) {
        $env:PROCESSOR_ARCHITEW6432
    } else {
        $env:PROCESSOR_ARCHITECTURE
    }
    if ($architecture -ne "AMD64") {
        Stop-Install "Windows $architecture release assets are not available"
    }
    return "windows-x64"
}

function Invoke-Download {
    param([string]$Uri, [string]$Destination)

    if (Test-Path -LiteralPath $Uri -PathType Leaf) {
        Copy-Item -LiteralPath $Uri -Destination $Destination
        return
    }
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination -Headers @{
            "User-Agent" = "OmniInfer-installer"
        }
    } catch {
        Stop-Install "failed to download ${Uri}: $($_.Exception.Message)"
    }
}

function Read-ExpectedChecksum {
    param([string]$ChecksumsPath, [string]$AssetName)

    foreach ($line in Get-Content -LiteralPath $ChecksumsPath) {
        if ($line -match '^\s*([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$' -and $Matches[2] -eq $AssetName) {
            return $Matches[1].ToLowerInvariant()
        }
    }
    Stop-Install "checksums.txt does not contain $AssetName"
}

function Add-InstallDirToPath {
    param([string]$Directory)

    $normalized = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $userEntries = @($userPath -split ';' | Where-Object { $_ })
    $alreadyInUserPath = $userEntries | Where-Object {
        ([IO.Path]::GetFullPath($_).TrimEnd('\')).Equals($normalized, [StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $alreadyInUserPath) {
        $newUserPath = (@($userEntries) + $normalized) -join ';'
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Ok "Added $normalized to the user PATH"
    }

    $processEntries = @($env:Path -split ';' | Where-Object { $_ })
    $alreadyInProcessPath = $processEntries | Where-Object {
        ([IO.Path]::GetFullPath($_).TrimEnd('\')).Equals($normalized, [StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $alreadyInProcessPath) {
        $env:Path = (@($processEntries) + $normalized) -join ';'
    }
}

if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    Stop-Install "invalid GitHub repository: $Repository"
}
if (-not $BaseUrl) {
    $BaseUrl = "https://github.com/$Repository/releases/download"
}

if ([Net.ServicePointManager]::SecurityProtocol -band [Net.SecurityProtocolType]::Tls12) {
    # TLS 1.2 is already enabled.
} else {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
}

$resolvedTarget = Resolve-Target
$runningOnWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
$resolvedVersion = Resolve-ReleaseVersion $Version
$assetName = "omniinfer-$resolvedVersion-$resolvedTarget.zip"
$releaseUrl = "$($BaseUrl.TrimEnd('/'))/$resolvedVersion"
$assetUrl = "$releaseUrl/$assetName"
$checksumsUrl = "$releaseUrl/checksums.txt"
$destinationExe = Join-Path $InstallDir "omniinfer.exe"

Write-Info "Version: $resolvedVersion"
Write-Info "Target: $resolvedTarget"
Write-Info "Install dir: $InstallDir"
Write-Info "Asset: $assetUrl"

if ($DryRun) {
    Write-Ok "Dry run complete"
    exit 0
}

$workDir = Join-Path ([IO.Path]::GetTempPath()) "omniinfer-install-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $workDir | Out-Null
try {
    $archivePath = Join-Path $workDir $assetName
    $checksumsPath = Join-Path $workDir "checksums.txt"
    $extractDir = Join-Path $workDir "extract"

    Write-Info "Downloading CLI archive"
    Invoke-Download $assetUrl $archivePath
    Write-Info "Downloading checksums"
    Invoke-Download $checksumsUrl $checksumsPath

    $expectedSha = Read-ExpectedChecksum $checksumsPath $assetName
    $actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($actualSha -ne $expectedSha) {
        Stop-Install "checksum mismatch for ${assetName}: expected $expectedSha, got $actualSha"
    }
    Write-Ok "Checksum verified: $actualSha"

    New-Item -ItemType Directory -Path $extractDir | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir
    $packageDir = Join-Path $extractDir "OmniInfer"
    $requiredFiles = @("omniinfer.exe", "omniinfer.cmd", "omniinfer.ps1")
    foreach ($name in $requiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $packageDir $name) -PathType Leaf)) {
            Stop-Install "archive did not contain OmniInfer/$name"
        }
    }

    if ($runningOnWindows) {
        $verifyOutput = & (Join-Path $packageDir "omniinfer.exe") --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Stop-Install "downloaded binary failed to run: $verifyOutput"
        }
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $transactionId = [Guid]::NewGuid().ToString("N")
    $staged = @{}
    foreach ($name in $requiredFiles) {
        $extension = [IO.Path]::GetExtension($name)
        $tempName = ".$([IO.Path]::GetFileNameWithoutExtension($name)).$transactionId$extension"
        $tempPath = Join-Path $InstallDir $tempName
        Copy-Item -LiteralPath (Join-Path $packageDir $name) -Destination $tempPath
        $staged[$name] = $tempPath
    }
    foreach ($name in @("omniinfer.cmd", "omniinfer.ps1", "omniinfer.exe")) {
        Move-Item -Force -LiteralPath $staged[$name] -Destination (Join-Path $InstallDir $name)
    }

    Write-Ok "Installed $destinationExe"
    if (-not $NoPathUpdate) {
        Add-InstallDirToPath $InstallDir
    } elseif (-not (($env:Path -split ';') -contains $InstallDir)) {
        Write-Warn "$InstallDir is not on PATH; add it to use omniinfer from a new shell"
    }

    if ($runningOnWindows) {
        $installedVersion = & $destinationExe --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Stop-Install "installed binary failed to run: $installedVersion"
        }
        Write-Host $installedVersion
    }
    Write-Ok "Next: run 'omniinfer backend list' and 'omniinfer backend install <backend>'"
} finally {
    if (Test-Path -LiteralPath $workDir) {
        Remove-Item -Recurse -Force -LiteralPath $workDir
    }
}
