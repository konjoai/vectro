#!/usr/bin/env python3
"""
validate_paper_results.py — Validates Vectro benchmark results against arXiv paper requirements

Performs comprehensive validation of:
1. Platform detection completeness
2. Benchmark result validity
3. Performance gate compliance
4. Statistical quality (CV <5%)
5. Accuracy contract validation
6. Paper table generation

Usage:
    python scripts/validate_paper_results.py [--results-dir PATH] [--strict]
    python scripts/validate_paper_results.py --results-dir benchmarks/results/cross_platform --strict
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


# Colors for output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"


def log_pass(msg: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def log_fail(msg: str):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def log_warn(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def log_info(msg: str):
    print(f"{Colors.BLUE}▶{Colors.RESET} {msg}")


def log_section(title: str):
    print(f"\n{Colors.BOLD}═══ {title} ═══{Colors.RESET}")


# ============================================================================
# Validation Gates (from ADR-002 & Paper Requirements)
# ============================================================================

VALIDATION_GATES = {
    "int8_min_throughput_vec_per_sec": 60_000,  # Python fallback floor
    "int8_min_throughput_m3_vec_per_sec": 10_000_000,  # ADR-002 M3 target
    "int8_min_quality_cosine": 0.9997,
    "nf4_min_quality_cosine": 0.9941,
    "binary_min_quality_cosine": 0.75,
    "adr002_max_latency_p99_ms": 1.0,
    "hnsw_min_recall_at_10": 0.90,
    "throughput_cv_percent_max": 5.0,
}

# ============================================================================
# Validation Functions
# ============================================================================


def validate_platform_metadata(platform: Dict) -> Tuple[bool, List[str]]:
    """Validate platform metadata completeness."""
    required_fields = [
        "os_type",
        "os_version",
        "architecture",
        "cpu_model",
        "cpu_cores",
        "cpu_frequency_ghz",
        "memory_gb",
        "simd_capabilities",
        "python_version",
        "numpy_version",
        "timestamp",
    ]

    issues = []
    for field in required_fields:
        if field not in platform:
            issues.append(f"Missing platform field: {field}")
        elif platform[field] is None:
            issues.append(f"Null platform field: {field}")

    return len(issues) == 0, issues


def validate_int8_throughput(results: List[Dict], cpu_arch: str) -> Tuple[bool, List[str]]:
    """Validate INT8 throughput results."""
    issues = []

    if not results:
        issues.append("No INT8 throughput results found")
        return False, issues

    # Determine appropriate floor based on architecture
    if "arm" in cpu_arch.lower():
        floor = VALIDATION_GATES["int8_min_throughput_m3_vec_per_sec"]
        arch_name = "M3"
    else:
        floor = VALIDATION_GATES["int8_min_throughput_vec_per_sec"]
        arch_name = "x86"

    for result in results:
        dim = result.get("dimension")
        mean_throughput = result.get("mean_vec_per_sec", 0)
        cv = result.get("cv_percent", 999)

        # Check throughput
        if mean_throughput < floor:
            issues.append(
                f"INT8 d={dim}: {mean_throughput:,.0f} vec/s < {floor:,} floor ({arch_name})"
            )

        # Check CV (statistical stability)
        if cv > VALIDATION_GATES["throughput_cv_percent_max"]:
            issues.append(
                f"INT8 d={dim}: CV={cv:.1f}% > {VALIDATION_GATES['throughput_cv_percent_max']}% target"
            )

    return len(issues) == 0, issues


def validate_quantization_quality(results: List[Dict], mode: str) -> Tuple[bool, List[str]]:
    """Validate quantization quality meets accuracy contracts."""
    issues = []

    if not results:
        return True, []  # Skip if no results

    gate_key = f"{mode}_min_quality_cosine"
    if gate_key not in VALIDATION_GATES:
        return True, []

    floor = VALIDATION_GATES[gate_key]

    for result in results:
        dim = result.get("dimension")
        mean_cosine = result.get("mean_cosine", 0)

        if mean_cosine < floor:
            issues.append(f"{mode.upper()} d={dim}: cosine={mean_cosine:.6f} < {floor:.6f} floor")

    return len(issues) == 0, issues


def validate_latency(results: Dict) -> Tuple[bool, List[str]]:
    """Validate single-vector latency meets ADR-002 target."""
    issues = []

    if not results:
        return True, []

    p99_latency = results.get("p99_ms", 999)
    floor = VALIDATION_GATES["adr002_max_latency_p99_ms"]

    if p99_latency > floor:
        issues.append(f"Latency p99={p99_latency:.4f}ms > {floor}ms (ADR-002 target)")

    return len(issues) == 0, issues


# ============================================================================
# Main Validation Logic
# ============================================================================


def validate_results(results_dir: Path, strict: bool = False) -> Tuple[bool, Dict]:
    """Perform comprehensive validation of benchmark results."""

    validation_report = {
        "timestamp": datetime.now().isoformat() + "Z",
        "strict_mode": strict,
        "platforms": [],
        "summary": {
            "total_platforms": 0,
            "passed_gates": 0,
            "failed_gates": 0,
            "warnings": 0,
        },
    }

    # Find all JSON result files
    result_files = sorted(results_dir.glob("vectro_benchmark_*.json"))

    if not result_files:
        log_fail(f"No benchmark results found in {results_dir}")
        return False, validation_report

    log_info(f"Found {len(result_files)} result files")

    for result_file in result_files:
        with open(result_file) as f:
            results = json.load(f)

        platform = results.get("platform", {})
        cpu_model = platform.get("cpu_model", "Unknown")

        log_section(f"Validating: {cpu_model}")

        platform_report = {"file": str(result_file), "cpu_model": cpu_model, "validations": {}}

        # 1. Platform metadata
        log_info("Checking platform metadata")
        valid, issues = validate_platform_metadata(platform)
        if valid:
            log_pass("Platform metadata complete")
            platform_report["validations"]["metadata"] = True
        else:
            log_fail("Platform metadata incomplete")
            for issue in issues:
                log_warn(f"  {issue}")
            platform_report["validations"]["metadata"] = False
            validation_report["summary"]["warnings"] += len(issues)

        # 2. INT8 throughput
        log_info("Checking INT8 throughput")
        int8_results = results.get("benchmarks", {}).get("int8_throughput", [])
        valid, issues = validate_int8_throughput(
            int8_results, platform.get("architecture", "unknown")
        )
        if valid:
            log_pass("INT8 throughput gates passed")
            platform_report["validations"]["int8_throughput"] = True
            validation_report["summary"]["passed_gates"] += 1
        else:
            log_fail("INT8 throughput gate failed")
            for issue in issues:
                log_warn(f"  {issue}")
            platform_report["validations"]["int8_throughput"] = False
            validation_report["summary"]["failed_gates"] += len(issues)

        # 3. NF4 quality
        log_info("Checking NF4 quality")
        nf4_results = results.get("benchmarks", {}).get("quality", [])
        nf4_quality = [r for r in nf4_results if r.get("mode") == "nf4"]
        valid, issues = validate_quantization_quality(nf4_quality, "nf4")
        if valid:
            log_pass("NF4 quality contract met")
            platform_report["validations"]["nf4_quality"] = True
        else:
            if strict:
                log_fail("NF4 quality gate failed (strict mode)")
                for issue in issues:
                    log_warn(f"  {issue}")
                validation_report["summary"]["failed_gates"] += len(issues)
                platform_report["validations"]["nf4_quality"] = False
            else:
                log_warn("NF4 quality gate failed (non-strict, warning only)")
                platform_report["validations"]["nf4_quality"] = False
                validation_report["summary"]["warnings"] += 1

        # 4. Latency
        log_info("Checking ADR-002 latency target")
        latency_results = results.get("benchmarks", {}).get("latency", [{}])[0]
        valid, issues = validate_latency(latency_results)
        if valid:
            log_pass("ADR-002 latency gate passed")
            platform_report["validations"]["adr002_latency"] = True
        else:
            log_fail("ADR-002 latency gate failed")
            for issue in issues:
                log_warn(f"  {issue}")
            platform_report["validations"]["adr002_latency"] = False
            validation_report["summary"]["failed_gates"] += len(issues)

        # 5. HNSW (if available)
        log_info("Checking HNSW recall")
        hnsw_results = results.get("benchmarks", {}).get("hnsw", [])
        if hnsw_results:
            for hnsw_result in hnsw_results:
                recall = hnsw_result.get("recall_at_10", 0)
                if recall >= VALIDATION_GATES["hnsw_min_recall_at_10"]:
                    log_pass(
                        f"HNSW R@10={recall:.4f} met {VALIDATION_GATES['hnsw_min_recall_at_10']:.2f} target"
                    )
                else:
                    log_warn(
                        f"HNSW R@10={recall:.4f} < {VALIDATION_GATES['hnsw_min_recall_at_10']:.2f}"
                    )
        else:
            log_warn("No HNSW results found")

        validation_report["platforms"].append(platform_report)
        validation_report["summary"]["total_platforms"] += 1

    return validation_report["summary"]["failed_gates"] == 0, validation_report


# ============================================================================
# Paper Table Generation
# ============================================================================


def generate_paper_tables(results_dir: Path) -> None:
    """Generate LaTeX-compatible paper tables from results."""

    log_section("Paper Table Generation")

    result_files = sorted(results_dir.glob("vectro_benchmark_*.json"))

    for result_file in result_files:
        with open(result_file) as f:
            results = json.load(f)

        platform = results.get("platform", {})
        cpu_model = platform.get("cpu_model", "Unknown").replace(",", "").split("(")[0].strip()

        # Table 1: INT8 Throughput
        int8_results = results.get("benchmarks", {}).get("int8_throughput", [])
        if int8_results:
            csv_file = results_dir / f"table1_int8_throughput_{cpu_model.replace(' ', '_')}.csv"

            with open(csv_file, "w") as f:
                f.write("Dimension,Mean (vec/s),Std Dev,CV (%)\n")
                for r in int8_results:
                    f.write(
                        f"{r['dimension']},{r['mean_vec_per_sec']:.0f},"
                        f"{r['std_vec_per_sec']:.0f},{r['cv_percent']:.1f}\n"
                    )

            log_pass(f"Generated: {csv_file.name}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate Vectro benchmark results against paper requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_paper_results.py
  python scripts/validate_paper_results.py --results-dir benchmarks/results/cross_platform
  python scripts/validate_paper_results.py --strict
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/cross_platform"),
        help="Directory containing benchmark results JSON files",
    )

    parser.add_argument(
        "--strict", action="store_true", help="Fail on any gate violation (otherwise warnings only)"
    )

    parser.add_argument(
        "--generate-tables", action="store_true", help="Generate LaTeX paper tables"
    )

    args = parser.parse_args()

    # Validate
    success, report = validate_results(args.results_dir, args.strict)

    # Print summary
    print()
    log_section("Validation Summary")
    print(f"Total platforms:  {report['summary']['total_platforms']}")
    print(f"Passed gates:     {report['summary']['passed_gates']}")
    print(f"Failed gates:     {report['summary']['failed_gates']}")
    print(f"Warnings:         {report['summary']['warnings']}")

    if success:
        log_pass("All validation gates passed ✓")
    else:
        if args.strict:
            log_fail("Validation failed in strict mode")
        else:
            log_warn("Some gates failed (non-strict mode, continuing)")

    # Generate tables
    if args.generate_tables:
        generate_paper_tables(args.results_dir)

    # Save report
    report_file = (
        args.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    log_pass(f"Report saved to: {report_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
