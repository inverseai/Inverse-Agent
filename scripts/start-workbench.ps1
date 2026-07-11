param(
    [string]$WorkspaceRoot = (Get-Location).Path,
    [int]$Port = 8765,
    [string]$ApproverId = "$([Environment]::UserName)@local",
    [switch]$SkipModelStart
)

$ErrorActionPreference = "Stop"

function New-SessionToken {
    param([int]$ByteCount = 32)

    $bytes = [byte[]]::new($ByteCount)
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Inverse-Agent virtual environment not found at $python. Run 'uv sync --locked --extra dev --python 3.12'."
}

$resolvedWorkspace = (Resolve-Path -LiteralPath $WorkspaceRoot -ErrorAction Stop).Path
if (-not $SkipModelStart) {
    & (Join-Path $PSScriptRoot "start-local-model.ps1")
}

$approvalSecret = $env:INVERSE_AGENT_APPROVAL_SECRET
if ([string]::IsNullOrWhiteSpace($approvalSecret) -or $approvalSecret.Length -lt 32) {
    $approvalSecret = [Environment]::GetEnvironmentVariable(
        "INVERSE_AGENT_APPROVAL_SECRET",
        [EnvironmentVariableTarget]::User
    )
}
if ([string]::IsNullOrWhiteSpace($approvalSecret) -or $approvalSecret.Length -lt 32) {
    $approvalSecret = New-SessionToken -ByteCount 48
    [Environment]::SetEnvironmentVariable(
        "INVERSE_AGENT_APPROVAL_SECRET",
        $approvalSecret,
        [EnvironmentVariableTarget]::User
    )
    Write-Output "Created a durable per-user approval signing secret."
}

$operatorToken = New-SessionToken
$approverToken = New-SessionToken
$env:INVERSE_AGENT_APPROVAL_SECRET = $approvalSecret
$env:INVERSE_AGENT_API_TOKEN = $operatorToken
$env:INVERSE_AGENT_APPROVER_TOKEN = $approverToken
$env:INVERSE_AGENT_APPROVER_ID = $ApproverId

Write-Output ""
Write-Output "Inverse-Agent engineering workbench"
Write-Output "URL: http://127.0.0.1:$Port"
Write-Output "Workspace root: $resolvedWorkspace"
Write-Output "Operator token: $operatorToken"
Write-Output "Approver token: $approverToken"
Write-Output "Approver identity: $ApproverId"
Write-Output "Keep this terminal open. Operator and approver tokens expire when the server stops."
Write-Output ""

& $python -m inverse_agent.cli serve --workspace-root $resolvedWorkspace --port $Port
exit $LASTEXITCODE
