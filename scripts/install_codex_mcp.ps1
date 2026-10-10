<# Register the local HiDream MCP server in the user's Codex configuration. #>

[CmdletBinding()]
param(
    [string]$Name = "hidream",
    [string]$ApiUrl = "http://127.0.0.1:7860",
    [string]$OutputDirectory,
    [ValidateRange(60, 7200)]
    [int]$ToolTimeoutSeconds = 1800,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$server = Join-Path $projectRoot "mcp_server.py"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot "results\mcp"
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment not found: $python"
}
if (-not (Test-Path -LiteralPath $server -PathType Leaf)) {
    throw "MCP server not found: $server"
}
$codex = Get-Command codex.cmd -ErrorAction SilentlyContinue
if (-not $codex) {
    $codex = Get-Command codex -ErrorAction SilentlyContinue
}
if (-not $codex) {
    throw "The codex CLI was not found on PATH."
}

& $python -c "import httpx, mcp, PIL" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "MCP dependencies are missing. Run: .\.venv\Scripts\python.exe -m pip install -r requirements-windows-cuda.txt"
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $codex.Source mcp get $Name 1> $null 2> $null
$exists = $LASTEXITCODE -eq 0
$ErrorActionPreference = $previousErrorActionPreference
if ($exists -and -not $Force) {
    throw "An MCP server named '$Name' already exists. Re-run with -Force to replace it."
}
if ($exists) {
    & $codex.Source mcp remove $Name
    if ($LASTEXITCODE -ne 0) { throw "Could not remove the existing '$Name' MCP entry." }
}

& $codex.Source mcp add $Name `
    --env "HIDREAM_API_URL=$ApiUrl" `
    --env "HIDREAM_MCP_OUTPUT_DIR=$OutputDirectory" `
    -- $python $server
if ($LASTEXITCODE -ne 0) {
    throw "Codex could not register the HiDream MCP server."
}

# Long GPU generations exceed Codex's default 60-second tool timeout. The CLI
# writes the server table; add the two timeout keys directly beneath its header.
$configPath = Join-Path $env:USERPROFILE ".codex\config.toml"
$configText = [System.IO.File]::ReadAllText($configPath)
$headerPattern = "(?m)^" + [regex]::Escape("[mcp_servers.$Name]") + "\r?$"
$headerRegex = [regex]::new($headerPattern)
if (-not $headerRegex.IsMatch($configText)) {
    throw "Codex registered '$Name', but its config table could not be located in $configPath"
}
$replacement = [System.Text.RegularExpressions.MatchEvaluator]{
    param($match)
    return $match.Value + "`r`nstartup_timeout_sec = 20`r`ntool_timeout_sec = $ToolTimeoutSeconds"
}
$configText = $headerRegex.Replace($configText, $replacement, 1)
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($configPath, $configText, $utf8NoBom)

Write-Host "Registered MCP server '$Name'."
Write-Host "API: $ApiUrl"
Write-Host "Outputs: $OutputDirectory"
Write-Host "Restart Codex, then run /mcp or ask it to call HiDream get_status."
