<# Install the portable HiDream skill without changing MCP configuration. #>

[CmdletBinding()]
param(
    [string]$SkillsDirectory,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $projectRoot "skills\hidream"
if ([string]::IsNullOrWhiteSpace($SkillsDirectory)) {
    $codexDirectory = $env:CODEX_HOME
    if ([string]::IsNullOrWhiteSpace($codexDirectory)) {
        $codexDirectory = Join-Path $env:USERPROFILE ".codex"
    }
    $SkillsDirectory = Join-Path $codexDirectory "skills"
}
$destination = [System.IO.Path]::GetFullPath((Join-Path $SkillsDirectory "hidream"))
$files = @("SKILL.md", "agents\openai.yaml", "references\setup.md")

# Preflight every file before changing the installed skill. Identical installs
# are safe to repeat; replacing local customizations requires an explicit flag.
foreach ($relativePath in $files) {
    $sourceFile = Join-Path $source $relativePath
    $targetFile = Join-Path $destination $relativePath
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "Skill source file not found: $sourceFile"
    }
    if (Test-Path -LiteralPath $targetFile -PathType Container) {
        throw "Expected a file, but found a directory: $targetFile"
    }
    if ((Test-Path -LiteralPath $targetFile -PathType Leaf) -and -not $Force) {
        if ((Get-FileHash -LiteralPath $sourceFile).Hash -ne (Get-FileHash -LiteralPath $targetFile).Hash) {
            throw "Installed skill differs at '$targetFile'. Use -Force to replace it."
        }
    }
}

foreach ($relativePath in $files) {
    $sourceFile = Join-Path $source $relativePath
    $targetFile = Join-Path $destination $relativePath
    if ([System.IO.Path]::GetFullPath($sourceFile) -eq $targetFile) { continue }
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $targetFile)) | Out-Null
    Copy-Item -LiteralPath $sourceFile -Destination $targetFile -Force
}

Write-Host "Installed HiDream skill: $destination"
Write-Host 'Automatic selection is enabled. You can also invoke it as $hidream.'
Write-Host "Reload or restart Codex if the skill does not appear."
Write-Host "MCP registration and the running image backend are separate prerequisites; see docs/MCP.md."
