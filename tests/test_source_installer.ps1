param(
    [string]$Installer = (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\install-from-source.ps1")
)

$ErrorActionPreference = "Stop"
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $Installer),
    [ref]$tokens,
    [ref]$errors
)
if ($errors.Count -gt 0) {
    throw "Source installer has PowerShell parse errors: $($errors -join '; ')"
}

$waitFunction = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Wait-ForExitKey"
}, $true)
if (-not $waitFunction) {
    throw "Wait-ForExitKey function was not found"
}

$fatalFunction = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Stop-Fatal"
}, $true)
if (-not $fatalFunction -or $fatalFunction.Extent.Text -notmatch '\bWait-ForExitKey\b') {
    throw "Stop-Fatal must use the shared non-interactive exit helper"
}

Invoke-Expression $waitFunction.Extent.Text
$NonInteractive = $true
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$output = & { Wait-ForExitKey } 2>&1 | Out-String
$timer.Stop()

if ($output.Trim()) {
    throw "Non-interactive failure path produced an exit prompt: $output"
}
if ($timer.Elapsed.TotalSeconds -ge 1) {
    throw "Non-interactive failure path waited for $($timer.Elapsed.TotalSeconds) seconds"
}

Write-Host "source installer PowerShell tests passed"
