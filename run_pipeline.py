"""
ParkIQ — Master Pipeline Runner (Enhanced)
Modules 0-7, each run as isolated subprocess.
Usage: python run_pipeline.py
"""
import subprocess, sys, time, os

MODULES = [
    ("Module 0 — Ingest & CIS scoring",          "module0_ingest.py"),
    ("Module 1 — LSTM Junction Stress",           "module1_junction_stress.py"),
    ("Module 2 — Spillover Graph",                "module2_spillover.py"),
    ("Module 3 — Enforcement + Multi-route",      "module3_enforcement.py"),
    ("Module 4 — XGBoost + SHAP Attribution",     "module4_attribution.py"),
    ("Module 5 — Policy Simulation",              "module5_policy_sim.py"),
    ("Module 6 — Anomaly + Repeat Hotspots",      "module6_anomaly_repeat.py"),
    ("Module 7 — Revenue & ROI Intelligence",     "module7_roi.py"),
]


def run_module(label, script):
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    start  = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  ERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed:.1f}s")


def main():
    total_start = time.time()
    print("=" * 62)
    print("  ParkIQ — Full Pipeline  (8 modules)")
    print("=" * 62)
    for label, script in MODULES:
        run_module(label, script)
    elapsed = time.time() - total_start
    print(f"\n{'='*62}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print("  All outputs in ./outputs/")
    print("  Now run:  streamlit run dashboard.py")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
