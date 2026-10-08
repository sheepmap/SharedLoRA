# Original evaluation backup

Snapshot of the src evaluation scripts before the October 8, 2026 statistics changes.

- evaluate.py reports mean +/- 2 * runstats sample standard deviation.
- The aggregated VAR is the population variance across acceleration-factor means.
- Current scripts in src report per-factor population variance and mean +/- population standard deviation.

These files are preserved for comparison. The shell script uses a relative evaluate.py path; do not run it from src expecting it to select this backup.
