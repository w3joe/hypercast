"""Verify the explicitly named Modal workspace before deploying the public demo.

Usage: python scripts/deploy_demo.py --profile hypercast-demo --workspace hypercast-demo
The CLI login and secret setup are described in docs/deployment.md.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--check", action="store_true", help="Check workspace identity without deploying")
    args = parser.parse_args()
    if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
        parser.error("Use the dedicated Modal profile; inherited MODAL_TOKEN_* variables override profiles.")
    command = [sys.executable, "-m", "modal"]
    env = {**os.environ, "NO_COLOR": "1"}
    check = subprocess.run([*command, "token", "info", "--profile", args.profile],
                           text=True, capture_output=True, env=env)
    match = re.search(r"Workspace:\s+([^\s]+)", check.stdout)
    if check.returncode or not match:
        parser.error("Could not verify the demo profile. Complete Modal token setup for the separate workspace first.")
    if match[1] != args.workspace:
        parser.error(f"Profile belongs to workspace {match[1]!r}, not {args.workspace!r}. Nothing was deployed.")
    print(f"Verified workspace: {args.workspace}")
    if not args.check:
        # No rolling overlap: one API container owns the SQLite volume.
        subprocess.run([*command, "deploy", "deploy/modal_demo.py", "--profile", args.profile,
                        "--env", "main", "--strategy", "recreate"],
                       cwd=Path(__file__).resolve().parents[1], env=env, check=True)


if __name__ == "__main__":
    main()
