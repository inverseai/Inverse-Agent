param(
    [string]$WorkspaceRoot = (Get-Location).Path,
    [int]$Port = 8765,
    [string]$ApproverId = "$([Environment]::UserName)@local",
    [ValidateSet(0, 16384, 24576, 32768, 49152)]
    [int]$ModelContextTokens = 0,
    [ValidateRange(0.0, 4.0)]
    [double]$ModelEstimatorBytesPerToken = 0.0,
    [ValidateSet("default", "low", "medium", "high")]
    [string]$ModelReasoningEffort = "default",
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
$hasContextCalibration = $ModelContextTokens -ne 0
$hasEstimatorCalibration = $ModelEstimatorBytesPerToken -ne 0.0
if ($hasContextCalibration -ne $hasEstimatorCalibration) {
    throw "ModelContextTokens and ModelEstimatorBytesPerToken must be supplied together."
}
if ($hasEstimatorCalibration -and $ModelEstimatorBytesPerToken -lt 1.0) {
    throw "ModelEstimatorBytesPerToken must be between 1.0 and 4.0."
}
if (-not $SkipModelStart) {
    if ($hasContextCalibration) {
        & (Join-Path $PSScriptRoot "start-local-model.ps1") `
            -ContextLength $ModelContextTokens
    }
    else {
        & (Join-Path $PSScriptRoot "start-local-model.ps1")
    }
}
if ($hasContextCalibration) {
    $env:INVERSE_AGENT_MODEL_CONTEXT_TOKENS = [string]$ModelContextTokens
    $env:INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN = `
        [string]$ModelEstimatorBytesPerToken
}
else {
    $env:INVERSE_AGENT_MODEL_CONTEXT_TOKENS = $null
    $env:INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN = $null
}
if ($ModelReasoningEffort -eq "default") {
    $env:INVERSE_AGENT_MODEL_REASONING_EFFORT = $null
}
else {
    $env:INVERSE_AGENT_MODEL_REASONING_EFFORT = $ModelReasoningEffort
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
