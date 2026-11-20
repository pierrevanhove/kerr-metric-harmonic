#!/usr/bin/env python3
"""
Run every check of the repository from the command line:

    python run_all_checks.py

Each module is run as a script; the exit status is 0 if all of them succeed.
The notebook checks.ipynb runs the same checks with explanations.
"""
import os
import subprocess
import sys
import time

MODULES = ["cubic_action_identities.py", "worldline_source_checks.py", "kerr_harmonic_checks.py",
           "kerr_relaxed_einstein_checks.py", "kerr_canonical_mpm_checks.py", "kerr_260519948_v1_4pm_checks.py"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    failed = []
    t0 = time.time()
    for mod in MODULES:
        print("=" * 100)
        print(mod)
        print("=" * 100, flush=True)
        if subprocess.call([sys.executable, os.path.join(here, mod)], cwd=here) != 0:
            failed.append(mod)
    print("=" * 100)
    print("all checks passed" if not failed else "FAILED: " + ", ".join(failed), "(%.0f s)" % (time.time() - t0))
    sys.exit(1 if failed else 0)
