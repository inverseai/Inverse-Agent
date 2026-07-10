param(
    [string]$ModelKey = "openai/gpt-oss-20b",
    [string]$Identifier = "inverse-gpt-oss-20b",
    [int]$ContextLength = 16384,
    [int]$Port = 1234
)

$lms = (Get-Command lms -ErrorAction SilentlyContinue).Source
if (-not $lms) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\LM Studio\resources\app\.webpack\lms.exe"
    if (Test-Path -LiteralPath $candidate) {
        $lms = $candidate
    }
}
if (-not $lms) {
    throw "LM Studio CLI was not found. Install LM Studio before running this script."
}

function Assert-LoopbackListener {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
    if (-not $listeners) {
        throw "LM Studio is not listening on port $Port."
    }
    $unsafe = @($listeners | Where-Object { $_.LocalAddress -notin @("127.0.0.1", "::1") })
    if ($unsafe) {
        $addresses = ($unsafe.LocalAddress | Sort-Object -Unique) -join ", "
        throw "LM Studio port $Port has non-loopback listeners: $addresses"
    }
    if ($listeners.LocalAddress -notcontains "127.0.0.1") {
        throw "LM Studio is not bound to 127.0.0.1 on port $Port."
    }
}

& $lms server start --port $Port --bind 127.0.0.1
if ($LASTEXITCODE -ne 0) {
    & $lms server status
    if ($LASTEXITCODE -ne 0) {
        throw "LM Studio server did not start."
    }
}
Assert-LoopbackListener

$loaded = (& $lms ps --json | Out-String)
if ($loaded -notmatch [regex]::Escape($Identifier)) {
    & $lms load $ModelKey --gpu max --context-length $ContextLength `
        --identifier $Identifier --ttl 3600 --yes
    if ($LASTEXITCODE -ne 0) {
        throw "LM Studio could not load $ModelKey."
    }
}

$env:INVERSE_AGENT_MODEL_NAME = $Identifier
$env:INVERSE_AGENT_MODEL_BASE_URL = "http://127.0.0.1:$Port/v1"

Write-Output "LM Studio is ready on $env:INVERSE_AGENT_MODEL_BASE_URL"
Write-Output "Model identifier: $env:INVERSE_AGENT_MODEL_NAME"
