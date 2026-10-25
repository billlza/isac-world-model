"""Check the actual native toolchain without installing or changing defaults."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def inspect(root: Path) -> dict:
    wanted = json.loads((root / "ci/versions.json").read_text())["rust"]["version"]
    tools = {}
    for name, args in [
        ("rustc", ["--version"]),
        ("cargo", ["--version"]),
        ("rustfmt", ["--version"]),
        ("cargo-clippy", ["--version"]),
        ("cc", ["--version"]),
    ]:
        executable = shutil.which(name)
        if not executable:
            raise FileNotFoundError(f"{name} is required")
        proc = subprocess.run(
            [executable, *args], capture_output=True, text=True, check=True, cwd=root, timeout=15
        )
        tools[name] = {"path": executable, "version": proc.stdout.splitlines()[0]}
    for name in ("rustc", "cargo"):
        if tools[name]["version"].split()[1] != wanted:
            raise ValueError(f"{name} is not pinned version {wanted}")
    return {"status": "passed", "expected_rust": wanted, "tools": tools}


if __name__ == "__main__":
    try:
        print(json.dumps(inspect(Path(__file__).resolve().parents[1]), indent=2))
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
