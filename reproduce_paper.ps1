# reproduce_paper.ps1 — Windows PowerShell equivalent of reproduce_paper.sh.
#
# Usage:
#   .\reproduce_paper.ps1 -Wave 1 -Runs 3 -Platform windows-x86_64 -Output results/paper
#
# Outputs the same JSON schema (vectro/paper/wave-bench/v2) so
# scripts/aggregate_paper_tables.py can mix Windows and POSIX results.

[CmdletBinding()]
param(
    [string]$Platform = "",
    [string]$Wave = "1",
    [int]$Runs = 3,
    [string]$Output = "results/paper",
    [switch]$Cold = $false,
    [switch]$ThermalAware = $false
)

$ErrorActionPreference = "Stop"

# ── 1. Platform ──────────────────────────────────────────────────
if (-not $Platform) {
    $arch = (Get-CimInstance Win32_Processor).Architecture
    $archName = switch ($arch) {
        9  { "x86_64" }
        12 { "arm64"  }
        default { "x86_64" }
    }
    $Platform = "windows-$archName"
}

# ── 2. Clean tree gate ───────────────────────────────────────────
$gitStatus = git status --porcelain 2>$null
if ($gitStatus) {
    Write-Error "ERROR: repo has uncommitted changes — refusing to record a benchmark"
    exit 3
}
$GitRev = (git rev-parse HEAD).Trim()

# ── 3. Output dir + filename ─────────────────────────────────────
$DateTag = Get-Date -Format "yyyyMMddTHHmmssZ" -AsUTC
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$OutFile = Join-Path $Output "wave${Wave}_${Platform}_${DateTag}.json"

# ── 4. Background-process check ──────────────────────────────────
# Windows lacks a single "load average" — proxy by sampling % CPU.
$cpuLoad = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 2 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty CounterSamples |
    Measure-Object -Property CookedValue -Average).Average
$LoadOk = if ($cpuLoad -lt 50) { 1 } else { 0 }
if (-not $LoadOk) {
    Write-Warning "CPU load ${cpuLoad}% — results may include noise"
}

# ── 5. Thread count pinning (physical cores) ─────────────────────
$PhysicalCores = (Get-CimInstance Win32_Processor |
    Measure-Object -Property NumberOfCores -Sum).Sum
if (-not $PhysicalCores) { $PhysicalCores = 1 }
$env:OMP_NUM_THREADS = $PhysicalCores
$env:RAYON_NUM_THREADS = $PhysicalCores

# ── 6. Thermal probe (best effort) ───────────────────────────────
function Read-Thermal {
    try {
        $temp = (Get-CimInstance -Namespace "root/wmi" -ClassName "MSAcpi_ThermalZoneTemperature" -ErrorAction SilentlyContinue |
            Select-Object -First 1).CurrentTemperature
        if ($temp) {
            # Tenths of Kelvin → Celsius
            return [math]::Round(($temp / 10 - 273.15), 1)
        }
    } catch {}
    return "unknown"
}
$ThermalBefore = Read-Thermal

# ── 7. SIMD probe via Rust bridge ────────────────────────────────
$Simd = & python -c "from python._rust_bridge import simd_tier; print(simd_tier())" 2>$null
if (-not $Simd) { $Simd = "unknown" }

# ── 8. Cache-drop proxy ──────────────────────────────────────────
function Drop-CachesProxy {
    # No sudo purge equivalent on Windows.  Force a CLR GC + close any
    # large NumPy arrays held by previous runs, as best-effort proxy.
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}

# ── 9. Bench loop ────────────────────────────────────────────────
$BenchCmd = @("python", "benchmarks/vectro_paper_benchmark.py", "--quick", "--table", "int8", "--json")
$Throughputs = @()
$attempt = 0
$maxAttempts = $Runs + 2

while ($attempt -lt $maxAttempts -and $Throughputs.Count -lt $Runs) {
    $attempt++
    if ($Cold) { Drop-CachesProxy }
    $raw = & $BenchCmd[0] $BenchCmd[1..($BenchCmd.Length - 1)] 2>$null
    if (-not $raw) { $raw = '{"throughput": 0}' }
    Write-Host "  run ${attempt}: $raw"
    try {
        $obj = $raw | ConvertFrom-Json
        if ($obj.throughput -is [double] -or $obj.throughput -is [int]) {
            $Throughputs += [double]$obj.throughput
        }
    } catch {
        # ignore parse failures; the run will simply not contribute a sample
    }
}

# ── 10. Statistics ───────────────────────────────────────────────
$Mean  = if ($Throughputs.Count -gt 0) { ($Throughputs | Measure-Object -Average).Average } else { $null }
$Stdev = 0.0
if ($Throughputs.Count -ge 2) {
    $sumSq = 0.0
    foreach ($x in $Throughputs) { $sumSq += [math]::Pow($x - $Mean, 2) }
    $Stdev = [math]::Sqrt($sumSq / $Throughputs.Count)
}
$CovPct = if ($Mean -and $Mean -ne 0) { [math]::Round(100 * $Stdev / $Mean, 3) } else { 0 }

$ThermalAfter = Read-Thermal

# ── 11. Runtime versions ─────────────────────────────────────────
$PyVer   = & python -c "import sys; print(sys.version.split()[0])"
$NpVer   = & python -c "import numpy; print(numpy.__version__)" 2>$null
if (-not $NpVer) { $NpVer = "unknown" }
$RustVer = (rustc --version) -join ""
if (-not $RustVer) { $RustVer = "unknown" }

# ── 12. Write JSON ───────────────────────────────────────────────
$record = [ordered]@{
    version          = "v2"
    schema           = "vectro/paper/wave-bench/v2"
    git_rev          = $GitRev
    platform         = $Platform
    wave             = [int]$Wave
    runs             = [int]$Runs
    cold             = [bool]$Cold
    thermal_aware    = [bool]$ThermalAware
    load_avg_ok      = [bool]$LoadOk
    load_avg         = $cpuLoad
    thermal_before   = $ThermalBefore
    thermal_after    = $ThermalAfter
    simd             = $Simd
    omp_threads      = [int]$env:OMP_NUM_THREADS
    rayon_threads    = [int]$env:RAYON_NUM_THREADS
    physical_cores   = [int]$PhysicalCores
    python           = $PyVer
    rustc            = $RustVer
    numpy            = $NpVer
    throughputs      = $Throughputs
    throughput_mean  = $Mean
    throughput_stdev = $Stdev
    cov_pct          = $CovPct
    timestamp_utc    = (Get-Date).ToUniversalTime().ToString("o")
}
$record | ConvertTo-Json -Depth 6 | Set-Content -Path $OutFile -Encoding UTF8

Write-Host ""
Write-Host "✓ wrote $OutFile"
Write-Host "  CoV: $CovPct%  (gate: 5%)"
if ($CovPct -gt 5.0) {
    Write-Warning "CoV exceeds 5% — investigate noise sources"
}
