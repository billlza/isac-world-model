"""Install the wheel into an empty target outside the checkout, offline.

Dependencies come from the tested host environment. This is NOT a claim of
reproducing all dependencies in a clean virtual machine.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dist", type=Path, required=True)
args = parser.parse_args()
wheels = list(args.dist.resolve().glob("isac_world_model-*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected exactly one built project wheel")
with tempfile.TemporaryDirectory(prefix="isac-wheel-") as temporary:
    root = Path(temporary)
    site = root / "site"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--disable-pip-version-check",
            "--target",
            str(site),
            str(wheels[0]),
        ],
        cwd=root,
        env=env,
        check=True,
        timeout=30,
    )
    env["PYTHONPATH"] = str(site)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import isac_wm,json; print(json.dumps({'path':isac_wm.__file__,'version':isac_wm.__version__}))",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(probe.stdout)
    if not Path(result["path"]).is_relative_to(root):
        raise AssertionError("Imported source checkout instead of installed wheel")
    subprocess.run([sys.executable, "-m", "isac_wm", "--help"], cwd=root, env=env, check=True, timeout=30)
    print("Installed-wheel import and CLI: PASS", result)
