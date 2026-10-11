[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Colibri\colibri",
    [string]$Model = "C:\Colibri\Models\Qwen3.8-Flash-Next-FP8",
    [int]$Cap = 28,
    [int]$Ctx = 8192
)

$ErrorActionPreference = "Stop"

$coli = Join-Path $RepoRoot "c\coli.cmd"
$engine = Join-Path $RepoRoot "c\qwen38.exe"
$sidecar = Join-Path $Model "q38-experts-q4g64.bin"

if (!(Test-Path $coli))   { throw "coli.cmd not found: $coli" }
if (!(Test-Path $engine)) { throw "qwen38.exe not found: $engine" }
if (!(Test-Path $Model))  { throw "Model folder not found: $Model" }
if (!(Test-Path $sidecar)){ throw "Q4 sidecar not found: $sidecar" }

# Model / launcher
$env:COLI_MODEL = $Model
$env:SNAP = $Model
$env:Q38_Q4_SIDECAR = $sidecar

# Champion CPU setup
$env:OMP_NUM_THREADS = "6"
$env:OMP_WAIT_POLICY = "PASSIVE"
$env:OMP_SCHEDULE = "dynamic,1"

$env:Q38_CACHE_POLICY = "lru"
$env:Q38_Q4_IO_MODE = "base"

$env:Q38_DN_AVX2 = "1"
$env:Q38_XF_SIMD_ACT = "1"
$env:Q38_XF_FUSED_DOWN = "1"
$env:Q38_CPU_V3 = "1"
$env:Q38_Q4_TLS_SCRATCH = "1"

$env:Q38_Q4_ASYNC_SHARED = "0"
$env:Q38_EXPERT_PREFETCH = "0"
$env:Q38_EXPERT_PARALLEL_READS = "1"
$env:Q38_Q4_ACT_I8 = "1"

$env:Q38_TRUNK_CPU_INT8 = "1"
$env:Q38_TRUNK_GPU = "0"
$env:COLI_CUDA = "0"

# Keep normal chat clean. These experiments test only getenv(), so remove them
# instead of setting them to "0".
Remove-Item Env:Q38_CACHE_TELEMETRY -ErrorAction SilentlyContinue
Remove-Item Env:Q38_Q4_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:COLI_TIMERS -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "BABA Qwen3.8 Champion" -ForegroundColor Cyan
Write-Host "Model : $Model"
Write-Host "Cache : $Cap experts/layer"
Write-Host "Ctx   : $Ctx"
Write-Host "CPU   : OMP=6 PASSIVE | LRU | Q4 base I/O | DeltaNet AVX2"
Write-Host ""

# Use COLI_MODEL rather than --model to avoid Windows drive-letter/path parsing
# trouble. --cap explicitly wins over the launcher's automatic cache planner.
Push-Location (Join-Path $RepoRoot "c")
try {
    & $coli chat --cap $Cap --ctx $Ctx
}
finally {
    Pop-Location
}
