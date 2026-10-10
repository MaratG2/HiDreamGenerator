<# Start the HiDream MCP stdio server for manual testing or another MCP host. #>

[CmdletBinding()]
param(
    [string]$ApiUrl = "http://127.0.0.1:7860",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "results\mcp")
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$server = Join-Path $PSScriptRoot "mcp_server.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment not found: $python"
}
if (-not (Test-Path -LiteralPath $server -PathType Leaf)) {
    throw "MCP server not found: $server"
}

$env:HIDREAM_API_URL = $ApiUrl
$env:HIDREAM_MCP_OUTPUT_DIR = $OutputDirectory
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $python $server
exit $LASTEXITCODE
