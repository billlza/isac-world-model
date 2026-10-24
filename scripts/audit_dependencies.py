"""Audit installed releases without silently skipping PyTorch backend wheels."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from packaging.version import Version


def validate_report(report: dict) -> list[str]:
    errors = []
    if not isinstance(report.get("dependencies"), list) or not report["dependencies"]:
        return ["Empty advisory response"]
    for dep in report["dependencies"]:
        if dep.get("skip_reason"):
            errors.append(f"Unaudited package {dep.get('name')}: {dep['skip_reason']}")
        if dep.get("vulns"):
            errors.append(f"Known advisories for {dep.get('name')}")
    return errors


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    requirements = []
    mappings = []
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name == "isac-world-model":
            continue
        version = Version(dist.version)
        public = version.public
        if version.local:
            if name != "torch" or not re.fullmatch(r"(cpu|cu[0-9]+)", version.local):
                raise ValueError(f"Unreviewed local-build mapping: {name}=={version}")
            mappings.append(
                {
                    "package": name,
                    "installed": str(version),
                    "audited_release": public,
                    "scope": "Upstream release advisories; backend-specific binary provenance is separate",
                }
            )
        requirements.append(f"{name}=={public}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="isac-audit-") as temp:
        req = Path(temp) / "requirements.txt"
        req.write_text("\n".join(sorted(set(requirements))) + "\n")
        code = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(req),
                "--no-deps",
                "--disable-pip",
                "--progress-spinner",
                "off",
                "--format",
                "json",
                "--output",
                str(args.out),
            ],
            timeout=150,
            check=False,
        ).returncode
    if not args.out.exists():
        print("Advisory service produced no report", file=sys.stderr)
        return 2
    report = json.loads(args.out.read_text())
    errors = validate_report(report)
    report["local_version_mappings"] = mappings
    report["scope"] = "Known upstream advisories at query time; not a source/binary security proof"
    report["status"] = "passed" if code == 0 and not errors else "failed"
    report["validation_errors"] = errors
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], "audited packages:", len(report.get("dependencies", [])), "mappings:", mappings)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
