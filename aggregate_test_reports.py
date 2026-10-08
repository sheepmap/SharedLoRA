"""Average per-factor metric means and variances across test-set reports."""

import argparse
import math
from pathlib import Path
import re
import tempfile


METRIC_NAMES = ("HFN", "MSE", "NMSE", "PSNR", "SSIM")
ROW_PATTERN = re.compile(r"^\[(?:(VAR)\]\[)?([^\]]+)\]\s+(.*)$")


def parse_metrics(payload, context):
    """Read the value after each '='; '+/-' values are not variances."""
    values = {}
    for name in METRIC_NAMES:
        # Word boundaries keep MSE from matching the suffix of NMSE.
        matches = re.findall(rf"\b{name}\s*=\s*(\S+)", payload)
        if len(matches) != 1:
            raise ValueError(f"{context}: expected exactly one {name} value")
        try:
            value = float(matches[0])
        except ValueError as exc:
            raise ValueError(f"{context}: invalid {name} value {matches[0]!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{context}: non-finite {name} value {matches[0]!r}")
        values[name] = value
    return values


def read_report(path, acc_factors):
    rows = {"mean": {}, "variance": {}}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = ROW_PATTERN.fullmatch(line.strip())
        if match is None:
            continue
        variance_tag, factor, payload = match.groups()
        if factor not in acc_factors:
            continue  # Includes the existing cross-factor [AVG] row.
        kind = "variance" if variance_tag else "mean"
        context = f"{path}:{line_number} [{kind}][{factor}]"
        if factor in rows[kind]:
            raise ValueError(f"{context}: duplicate row")
        rows[kind][factor] = parse_metrics(payload, context)

    for factor in acc_factors:
        for kind in rows:
            if factor not in rows[kind]:
                raise ValueError(f"{path}: missing {kind} row for {factor}")
    return rows


def aggregate_reports(report_dir, report_prefix, tests, acc_factors):
    if not tests or not acc_factors:
        raise ValueError("tests and acc_factors must not be empty")
    if len(set(tests)) != len(tests) or len(set(acc_factors)) != len(acc_factors):
        raise ValueError("tests and acc_factors must not contain duplicates")

    reports = [
        read_report(report_dir / f"{report_prefix}_{test}.txt", acc_factors)
        for test in tests
    ]
    lines = [
        "# Test sets: " + " ".join(tests),
        f"# Equal weights: 1/{len(tests)} per test set (no sample-count weighting).",
        "# MEAN: arithmetic average of each test's metric mean.",
        "# MEAN_VAR: arithmetic average of each test's [VAR] value; not variance of means.",
        "# Precision is limited by the rounded values in the input reports.",
    ]
    for factor in acc_factors:
        for kind, tag in (("mean", "MEAN"), ("variance", "MEAN_VAR")):
            values = {
                name: math.fsum(report[kind][factor][name] / len(reports)
                                for report in reports)
                for name in METRIC_NAMES
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite aggregate for {kind} / {factor}")
            metrics = " ".join(f"{name} = {values[name]:.10g}" for name in METRIC_NAMES)
            lines.append(f"[{tag}][{factor}]   {metrics}")
    return "\n".join(lines) + "\n"


def write_summary(output_path, content):
    """Publish a complete summary atomically, after all reports pass validation."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=output_path.parent, prefix=output_path.name + ".", suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--report-prefix", required=True,
                        help="Filename prefix without _testN.txt")
    parser.add_argument("--tests", nargs="+", required=True)
    parser.add_argument("--acc-factors", nargs="+", required=True)
    args = parser.parse_args()
    output_path = args.report_dir / f"{args.report_prefix}_tests_avg.txt"
    try:
        content = aggregate_reports(
            args.report_dir, args.report_prefix, args.tests, args.acc_factors
        )
        write_summary(output_path, content)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(1, f"Aggregation failed: {exc}\n")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
