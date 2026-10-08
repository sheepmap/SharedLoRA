import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from aggregate_test_reports import METRIC_NAMES, aggregate_reports, parse_metrics

TESTS = [f"test{i}" for i in range(1, 6)]
FACTORS = ["1x", "2x", "4x", "8x", "10x"]
PREFIX = "report_fastmri_knee_gaussian_2x,8x,10x"
BASH = os.environ.get("REPORT_TEST_BASH")


def metric_values(test_number, factor_number, variance=False):
    scale = 0.001 if variance else 0.1
    return {
        name: test_number * factor_number * index * scale
        for index, name in enumerate(METRIC_NAMES, 1)
    }


def report_content(test_number):
    lines = []
    for factor_number, factor in enumerate(FACTORS, 1):
        means = " ".join(
            f"{name} = {value:.8e} +/- 999"
            for name, value in metric_values(test_number, factor_number).items()
        )
        variances = " ".join(
            f"{name} = {value:.8e}"
            for name, value in metric_values(test_number, factor_number, True).items()
        )
        lines.extend((f"[{factor}]   {means}", f"[VAR][{factor}]   {variances}"))
    # Must never be confused with a per-factor mean.
    lines.append("[AVG]   " + " ".join(f"{name} = 999" for name in METRIC_NAMES))
    return "\n".join(lines) + "\n"


class AggregateReportsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for number, test in enumerate(TESTS, 1):
            (self.root / f"{PREFIX}_{test}.txt").write_text(
                report_content(number), encoding="utf-8"
            )

    def cli(self):
        return subprocess.run(
            [sys.executable, str(SRC_DIR / "aggregate_test_reports.py"),
             "--report-dir", str(self.root), "--report-prefix", PREFIX,
             "--tests", *TESTS, "--acc-factors", *FACTORS],
            capture_output=True, text=True,
        )

    def test_equal_means_and_mean_variances_for_every_factor(self):
        summary = aggregate_reports(self.root, PREFIX, TESTS, FACTORS)
        self.assertIn("# Test sets: " + " ".join(TESTS), summary)
        rows = [line for line in summary.splitlines() if line.startswith("[")]
        self.assertEqual(len(rows), 10)
        for factor_number, factor in enumerate(FACTORS, 1):
            for variance, tag in ((False, "MEAN"), (True, "MEAN_VAR")):
                row = next(line for line in rows if line.startswith(f"[{tag}][{factor}]"))
                actual = parse_metrics(row, "summary")
                expected = metric_values(3, factor_number, variance)
                for name in METRIC_NAMES:
                    self.assertAlmostEqual(actual[name], expected[name])

    def test_mse_does_not_match_nmse_and_plus_minus_is_ignored(self):
        values = parse_metrics(
            "NMSE = 2e-3 +/- 999 MSE = 1e-4 +/- 888 "
            "HFN = .5 PSNR = +3.2E+1 SSIM = 9e-1", "fixture"
        )
        self.assertEqual(values["MSE"], 0.0001)
        self.assertEqual(values["NMSE"], 0.002)
        self.assertEqual(values["PSNR"], 32)

    def test_invalid_report_never_creates_summary(self):
        path = self.root / f"{PREFIX}_test5.txt"
        original = path.read_text(encoding="utf-8")
        cases = {
            "missing factor": "\n".join(
                line for line in original.splitlines() if not line.startswith("[10x]")
            ),
            "missing variance": "\n".join(
                line for line in original.splitlines() if not line.startswith("[VAR][10x]")
            ),
            "missing metric": original.replace("HFN =", "OTHER =", 1),
            "duplicate row": original + original.splitlines()[0] + "\n",
        }
        for invalid in ("nan", "inf", "-inf", "1e999", "bad", "1e-3oops"):
            cases[invalid] = original.replace("HFN = 5.00000000e-01", f"HFN = {invalid}", 1)
        for label, content in cases.items():
            with self.subTest(label=label):
                path.write_text(content, encoding="utf-8")
                result = self.cli()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Aggregation failed:", result.stderr)
                self.assertFalse((self.root / f"{PREFIX}_tests_avg.txt").exists())

    def test_missing_report(self):
        (self.root / f"{PREFIX}_test5.txt").unlink()
        self.assertNotEqual(self.cli().returncode, 0)
        self.assertFalse((self.root / f"{PREFIX}_tests_avg.txt").exists())

    def test_repeated_run_overwrites_summary_without_extra_rows(self):
        self.assertEqual(self.cli().returncode, 0)
        output = self.root / f"{PREFIX}_tests_avg.txt"
        first = output.read_text(encoding="utf-8")
        self.assertEqual(self.cli().returncode, 0)
        self.assertEqual(output.read_text(encoding="utf-8"), first)
        self.assertEqual(len(list(self.root.glob("*.txt"))), 6)

    def test_duplicate_test_or_factor_is_rejected(self):
        for tests, factors in ((TESTS + ["test1"], FACTORS), (TESTS, FACTORS + ["1x"])):
            with self.subTest(tests=tests, factors=factors):
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    aggregate_reports(self.root, PREFIX, tests, factors)


@unittest.skipUnless(BASH, "Set REPORT_TEST_BASH to a Bash executable for batch tests")
class BatchScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "data with spaces"
        self.run_dir = self.base / "experiments/fastmri_knee/gaussian/acc_2x,8x,10x/fastmritest_seed45"
        self.targets = self.base / "datasets/fastmri_knee/gaussian/test"
        self.code = self.root / "scripts"
        self.code.mkdir()
        for test in TESTS:
            target = self.targets / test
            target.mkdir(parents=True)
            (target / f"{test} volume.h5").touch()
        for factor in FACTORS:
            predictions = self.run_dir / f"results_{factor}"
            predictions.mkdir(parents=True)
            for test in TESTS:
                (predictions / f"{test} volume.h5").touch()
        script = (SRC_DIR / "evaluate_combinedall.sh").read_text(encoding="utf-8")
        script = script.replace("BASE_PATH='/root/autodl-tmp'",
                                "BASE_PATH=" + shlex.quote(self.base.as_posix()))
        self.script = self.code / "evaluate_combinedall.sh"
        self.script.write_text(script, encoding="utf-8", newline="\n")
        shutil.copyfile(SRC_DIR / "aggregate_test_reports.py", self.code / "aggregate_test_reports.py")
        # Replace only the costly HDF5 evaluator in the isolated fixture.
        (self.code / "evaluate.py").write_text(
            "import argparse, os\nfrom pathlib import Path\n"
            "p = argparse.ArgumentParser()\n"
            "for name in ('target-path', 'predictions-path', 'report-path', 'acc-factor', "
            "'report-file-acc-factor', 'mask-type', 'dataset-type'):\n"
            "    p.add_argument('--' + name)\n"
            "a = p.parse_args()\n"
            "root = Path(a.report_path)\n"
            "with (root / 'calls.log').open('a') as f: f.write(a.report_file_acc_factor + ' ' + a.acc_factor + '\\n')\n"
            "if os.environ.get('REPORT_TEST_FAIL') == a.acc_factor: raise SystemExit(7)\n"
            "report = root / f'report_{a.dataset_type}_{a.mask_type}_{a.report_file_acc_factor}.txt'\n"
            "n = int(Path(a.target_path).name[-1])\n"
            "names = ('HFN', 'MSE', 'NMSE', 'PSNR', 'SSIM')\n"
            "means = ' '.join(f'{name} = {n} +/- 999' for name in names)\n"
            "variances = ' '.join(f'{name} = {n * 10}' for name in names)\n"
            "with report.open('a') as f:\n"
            "    f.write(f'[{a.acc_factor}]   {means}\\n[VAR][{a.acc_factor}]   {variances}\\n')\n",
            encoding="utf-8",
        )

    def run_batch(self, fail_factor=None):
        env = os.environ.copy()
        env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
        if fail_factor:
            env["REPORT_TEST_FAIL"] = fail_factor
        command = "python() { " + shlex.quote(Path(sys.executable).as_posix()) + ' "$@"; }\n'
        command += "source " + shlex.quote(self.script.as_posix())
        return subprocess.run([BASH, "-c", command], cwd=self.root, env=env,
                              capture_output=True, text=True)

    def test_25_evaluations_six_reports_and_fresh_repeated_run(self):
        result = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.run_dir / "calls.log").read_text().splitlines()), 25)
        self.assertEqual(len(list(self.run_dir.glob("*.txt"))), 6)
        summary = self.run_dir / f"{PREFIX}_tests_avg.txt"
        first = summary.read_text()
        self.assertIn("[MEAN][1x]   HFN = 3", first)
        self.assertIn("[MEAN_VAR][1x]   HFN = 30", first)
        report = self.run_dir / f"{PREFIX}_test1.txt"
        report.write_text(report.read_text() + "[old_factor] stale\n")
        result = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary.read_text(), first)
        self.assertNotIn("stale", report.read_text())
        for test in TESTS:
            lines = (self.run_dir / f"{PREFIX}_{test}.txt").read_text().splitlines()
            self.assertEqual(len(lines), 10)

    def test_preflight_missing_prediction_or_empty_test_starts_no_evaluation(self):
        missing = self.run_dir / "results_10x/test5 volume.h5"
        missing.unlink()
        result = self.run_batch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing prediction", result.stderr)
        self.assertFalse((self.run_dir / "calls.log").exists())
        missing.touch()
        (self.targets / "test5/test5 volume.h5").unlink()
        result = self.run_batch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No .h5 files", result.stderr)
        self.assertFalse((self.run_dir / "calls.log").exists())

    def test_evaluation_failure_stops_and_removes_old_summary(self):
        summary = self.run_dir / f"{PREFIX}_tests_avg.txt"
        summary.write_text("old summary")
        result = self.run_batch(fail_factor="4x")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len((self.run_dir / "calls.log").read_text().splitlines()), 3)
        self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main()
