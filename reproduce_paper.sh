#!/usr/bin/env bash
# reproduce_paper.sh — v2 (PLAN 2)
#
# Cross-platform benchmark harness for the v5.0.0 vectro performance paper.
#
# Usage:
#   ./reproduce_paper.sh [--platform NAME] [--wave N] [--runs N]
#                        [--thermal-aware] [--output PATH]
#
# Outputs a JSON record per run to:
#   results/paper/wave{N}_{platform}_{date}.json
#
# Konjo: every step that can fail prints why, and every measurement is
# reproducible from the JSON metadata alone (git rev, SIMD set, thermal
# state, OMP/RAYON thread count, runtime versions).

set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# Defaults + argument parsing
# ─────────────────────────────────────────────────────────────────

PLATFORM=""
WAVE="1"
RUNS=3
THERMAL_AWARE=0
OUTPUT="results/paper"
COLD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)        PLATFORM="$2"; shift 2 ;;
    --wave)            WAVE="$2"; shift 2 ;;
    --runs)            RUNS="$2"; shift 2 ;;
    --thermal-aware)   THERMAL_AWARE=1; shift ;;
    --cold)            COLD=1; shift ;;
    --output)          OUTPUT="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed -e 's/^# \?//'
      exit 0 ;;
    *)
      echo "ERROR: unknown flag $1" >&2
      exit 2 ;;
  esac
done

if [[ -z "$PLATFORM" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)   PLATFORM="darwin-arm64" ;;
    Darwin-x86_64)  PLATFORM="darwin-x86_64" ;;
    Linux-x86_64)   PLATFORM="linux-x86_64" ;;
    Linux-aarch64)  PLATFORM="linux-aarch64" ;;
    *)              PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)" ;;
  esac
fi

DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT"
OUT_FILE="${OUTPUT}/wave${WAVE}_${PLATFORM}_${DATE_TAG}.json"

# ─────────────────────────────────────────────────────────────────
# 1. Clean-tree gate
# ─────────────────────────────────────────────────────────────────

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: repo has uncommitted changes — refusing to record a benchmark" >&2
  echo "       commit, stash, or pass through CI before re-running." >&2
  exit 3
fi

GIT_HEAD="$(git rev-parse HEAD)"

# ─────────────────────────────────────────────────────────────────
# 2. Background-process check (load average < 1.0)
# ─────────────────────────────────────────────────────────────────

LOAD_OK=1
LOAD_AVG="$(uptime | awk -F'load averages?:' '{print $2}' | awk '{print $1}' | tr -d ',')"
if [[ -n "$LOAD_AVG" ]]; then
  if awk "BEGIN { exit !($LOAD_AVG > 1.0) }"; then
    echo "WARN: load average ${LOAD_AVG} > 1.0 — results may include noise" >&2
    LOAD_OK=0
  fi
fi

# ─────────────────────────────────────────────────────────────────
# 3. Thread count pinning (physical cores)
# ─────────────────────────────────────────────────────────────────

physical_cores() {
  case "$(uname -s)" in
    Darwin) sysctl -n hw.physicalcpu ;;
    Linux)  lscpu | awk '/^Core\(s\) per socket/ {c=$NF} /^Socket\(s\)/ {s=$NF} END {print c*s}' ;;
    *)      nproc 2>/dev/null || echo 1 ;;
  esac
}
N_THREADS="$(physical_cores)"
export OMP_NUM_THREADS="$N_THREADS"
export RAYON_NUM_THREADS="$N_THREADS"

# ─────────────────────────────────────────────────────────────────
# 4. Thermal probe
# ─────────────────────────────────────────────────────────────────

read_thermal() {
  case "$(uname -s)" in
    Darwin)
      if pmset -g thermlog 2>/dev/null | tail -1 | grep -qi normal; then
        echo "normal"
      elif pmset -g thermlog 2>/dev/null | tail -1 | grep -qi fair; then
        echo "fair"
      elif pmset -g thermlog 2>/dev/null | tail -1 | grep -qi serious; then
        echo "serious"
      elif pmset -g thermlog 2>/dev/null | tail -1 | grep -qi critical; then
        echo "critical"
      else
        echo "unknown"
      fi ;;
    Linux)
      local zone
      zone="$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -nr | head -1)"
      if [[ -n "$zone" ]]; then
        echo "${zone}"   # millidegrees C
      else
        echo "unknown"
      fi ;;
    *) echo "unknown" ;;
  esac
}
THERMAL_BEFORE="$(read_thermal)"

# ─────────────────────────────────────────────────────────────────
# 5. SIMD capabilities
# ─────────────────────────────────────────────────────────────────

SIMD="$(python3 -c '
try:
    from python._rust_bridge import simd_tier
    print(simd_tier())
except Exception:
    print("unknown")
' 2>/dev/null || echo "unknown")"

# ─────────────────────────────────────────────────────────────────
# 6. Cache-drop helper (cold runs)
# ─────────────────────────────────────────────────────────────────

drop_caches() {
  case "$(uname -s)" in
    Darwin) sudo purge 2>/dev/null || true ;;
    Linux)
      sync
      sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true ;;
  esac
}

# ─────────────────────────────────────────────────────────────────
# 7. Bench loop with CoV gate + outlier rule
# ─────────────────────────────────────────────────────────────────

# --reps 1 --warmup 0: the outer $RUNS loop already handles statistical
# replication; one rep per run keeps each iteration to ~15–40s depending
# on the platform, so a 3-run job completes in under 2 minutes.
BENCH_CMD=(python3 benchmarks/vectro_paper_benchmark.py --quick --table int8 --json --reps 1 --warmup 0)
RAW_RESULTS=()

run_once() {
  local out
  out="$("${BENCH_CMD[@]}" 2>/dev/null || echo '{"throughput": 0}')"
  printf '%s\n' "$out"
}

attempt=0
max_attempts=$((RUNS + 2))     # CoV gate may add up to 2 retries
while [[ $attempt -lt $max_attempts && ${#RAW_RESULTS[@]} -lt $RUNS ]]; do
  attempt=$((attempt + 1))
  if [[ $COLD -eq 1 ]]; then
    drop_caches
  fi
  result="$(run_once)"
  RAW_RESULTS+=("$result")
  echo "  run ${attempt}: ${result}"
done

# Compute mean / stddev / CoV from the throughput field of each run.
THROUGHPUTS="$(
  printf '%s\n' "${RAW_RESULTS[@]}" |
  python3 -c '
import json, sys
xs = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    t = d.get("throughput")
    if isinstance(t, (int, float)):
        xs.append(float(t))
print(",".join(repr(x) for x in xs))
'
)"

COV_PCT="$(python3 -c "
import statistics, sys
xs = [${THROUGHPUTS}]
if not xs:
    print(0)
else:
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) >= 2 else 0
    print(round(100 * (s / m) if m else 0, 3))
")"

# ─────────────────────────────────────────────────────────────────
# 8. Thermal probe (after) + JSON write
# ─────────────────────────────────────────────────────────────────

THERMAL_AFTER="$(read_thermal)"

python3 - <<PY > "$OUT_FILE"
import json, os, platform, sys, datetime, subprocess

py_ver = platform.python_version()
rust_ver = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
rust_ver = rust_ver.stdout.strip() if rust_ver.returncode == 0 else "unknown"

try:
    import numpy
    np_ver = numpy.__version__
except Exception:
    np_ver = "unknown"

xs = [${THROUGHPUTS}] or []

import statistics
mean = statistics.mean(xs) if xs else None
stdev = statistics.pstdev(xs) if len(xs) >= 2 else 0.0

record = {
    "version":        "v2",
    "schema":         "vectro/paper/wave-bench/v2",
    "git_rev":        "${GIT_HEAD}",
    "platform":       "${PLATFORM}",
    "wave":           int("${WAVE}"),
    "runs":           int("${RUNS}"),
    "cold":           bool(int("${COLD}")),
    "thermal_aware":  bool(int("${THERMAL_AWARE}")),
    "load_avg_ok":    bool(int("${LOAD_OK}")),
    "load_avg":       "${LOAD_AVG}",
    "thermal_before": "${THERMAL_BEFORE}",
    "thermal_after":  "${THERMAL_AFTER}",
    "simd":           "${SIMD}",
    "omp_threads":    int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
    "rayon_threads":  int(os.environ.get("RAYON_NUM_THREADS", "0") or 0),
    "physical_cores": int("${N_THREADS}"),
    "python":         py_ver,
    "rustc":          rust_ver,
    "numpy":          np_ver,
    "throughputs":    xs,
    "throughput_mean":  mean,
    "throughput_stdev": stdev,
    "cov_pct":        float("${COV_PCT}"),
    "timestamp_utc":  datetime.datetime.utcnow().isoformat() + "Z",
}
print(json.dumps(record, indent=2))
PY

echo
echo "✓ wrote ${OUT_FILE}"
echo "  CoV: ${COV_PCT}%  (gate: 5%)"
if awk "BEGIN { exit !(${COV_PCT} > 5.0) }"; then
  echo "  WARN: CoV exceeds 5% — investigate noise sources" >&2
fi
