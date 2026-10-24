"""Version policy, environment inventory and explicit latest-release checks.

The default command is offline. Online checks are read-only and never install
packages or execute registry content. A missing package/tool is not a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE = 16 * 1024 * 1024


def load_manifest(root: Path = ROOT) -> dict:
    data = json.loads((root / "ci/versions.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported version manifest")
    for name, version in data["packages"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Invalid package name")
        parsed = Version(version)
        if parsed.is_prerelease or parsed.is_devrelease:
            raise ValueError(f"Non-stable pin: {name}=={version}")
    return data


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def matches(installed: str, required: str) -> bool:
    try:
        # A CPU wheel's +cpu suffix identifies its backend, not an older release.
        return Version(installed) in SpecifierSet(f"=={required}")
    except InvalidVersion:
        return False


def check_policy(root: Path = ROOT) -> list[str]:
    m = load_manifest(root)
    p = tomllib.loads((root / "pyproject.toml").read_text())
    expected = {canonical(k): v for k, v in m["packages"].items()}
    requirements = (
        p["project"]["dependencies"]
        + p["project"]["optional-dependencies"]["dev"]
        + p["build-system"]["requires"]
    )
    errors = []
    for text in requirements:
        r = Requirement(text)
        wanted = expected.get(canonical(r.name))
        if wanted is None or str(r.specifier) != f"=={wanted}":
            errors.append(f"Unpinned or mismatched direct dependency: {text}")
    constraints = {}
    for line in (root / "ci/constraints.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            r = Requirement(line)
            constraints[canonical(r.name)] = str(r.specifier)
    if constraints != {k: f"=={v}" for k, v in expected.items()}:
        errors.append("constraints.txt differs from versions.json")
    rust = tomllib.loads((root / "rust-toolchain.toml").read_text())["toolchain"]
    if rust["channel"] != m["rust"]["version"]:
        errors.append("Rust toolchain version mismatch")
    if not set(m["rust"]["components"]).issubset(rust.get("components", [])):
        errors.append("Missing Rust components")
    if (root / ".python-version").read_text().strip() != m["python"]["preferred"]:
        errors.append("Python default version mismatch")
    if p.get("tool", {}).get("uv", {}).get("required-version") != f"=={expected['uv']}":
        errors.append("uv version is not locked")
    if "3.15" in m["python"]["supported_minors"]:
        errors.append("Python 3.15 needs an explicit torch compatibility review")
    return errors


def environment(manifest: dict | None = None) -> dict:
    m = manifest or load_manifest()
    rows = []
    for name, wanted in m["packages"].items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = None
        rows.append(
            {
                "package": name,
                "expected": wanted,
                "installed": found,
                "status": "passed" if found and matches(found, wanted) else "failed",
            }
        )
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    supported = minor in m["python"]["supported_minors"]
    return {
        "status": "passed" if supported and all(r["status"] == "passed" for r in rows) else "failed",
        "python": platform.python_version(),
        "python_supported": supported,
        "python_preferred": m["python"]["preferred"],
        "packages": rows,
    }


def fetch_json(url: str) -> dict:
    if not url.startswith("https://pypi.org/pypi/"):
        raise ValueError("Only official PyPI release metadata is allowed")
    request = urllib.request.Request(url, headers={"User-Agent": "isac-wm-version-audit/0.4"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if not response.url.startswith("https://pypi.org/"):
            raise ValueError("Unapproved registry redirect")
        content = response.read(MAX_RESPONSE + 1)
    if len(content) > MAX_RESPONSE:
        raise ValueError("Registry response exceeds its size budget")
    return json.loads(content)


def latest(manifest: dict | None = None) -> dict:
    m = manifest or load_manifest()
    rows = []
    for name, pin in {**m["packages"], **m.get("optional_packages", {})}.items():
        url = f"https://pypi.org/pypi/{name}/json"
        try:
            data = fetch_json(url)
            version = str(data["info"]["version"])
            parsed = Version(version)
            files = data.get("urls", [])
            if (
                parsed.is_prerelease
                or parsed.is_devrelease
                or not any(not f.get("yanked", False) for f in files)
            ):
                raise ValueError("Registry latest is not a non-yanked stable release")
            status = "current" if matches(version, pin) else "update_available"
            rows.append(
                {
                    "package": name,
                    "pinned": pin,
                    "latest": version,
                    "status": status,
                    "requires_python": data["info"].get("requires_python"),
                    "source": url,
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            rows.append({"package": name, "pinned": pin, "status": "blocked", "error": str(exc)})
    status = (
        "blocked"
        if any(r["status"] == "blocked" for r in rows)
        else "updates_available"
        if any(r["status"] == "update_available" for r in rows)
        else "current"
    )
    return {"status": status, "checked_utc": datetime.now(timezone.utc).isoformat(), "packages": rows}


def snapshot() -> dict:
    packages = []
    for dist in metadata.distributions():
        packages.append(
            {
                "name": dist.metadata.get("Name", ""),
                "version": dist.version,
                "requires_python": dist.metadata.get("Requires-Python"),
                "requires_dist": dist.requires or [],
            }
        )
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(packages, key=lambda x: canonical(x["name"])),
        "manifest_sha256": hashlib.sha256((ROOT / "ci/versions.json").read_bytes()).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "environment", "latest", "snapshot"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        errors = check_policy()
        report = {"status": "failed" if errors else "passed", "errors": errors}
    elif args.command == "environment":
        report = environment()
    elif args.command == "latest":
        report = latest()
    else:
        report = snapshot()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    status = report.get("status", "passed")
    return 2 if status == "blocked" else 1 if status in {"failed", "updates_available"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
