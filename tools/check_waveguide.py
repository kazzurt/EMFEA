"""Bounded smoke gate for the synthetic 2.45 GHz WR340 benchmark."""
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["max_abs_power_balance_error"] < 0.005, report
assert report["max_abs_S21_phase_error_deg"] < 1.0, report
print("WR340 smoke: power balance <0.5%, phase error <1 degree")
