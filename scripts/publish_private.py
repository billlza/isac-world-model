"""First private GitHub publication with exact-commit receipts and a fail-closed CI wait.

No credentials are read or printed by this script. gh uses the user's local login.
A successful upload is not a successful CI run. Existing different history is never
replaced. Run without --execute for a local, offline preflight.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from evidence import TOP_LEVEL, verify

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = frozenset(
    (
        *TOP_LEVEL,
        ".github/workflows/tests.yml",
        "scripts/bootstrap.sh",
        "scripts/ci.py",
        "scripts/evidence.py",
        "ci/versions.json",
    )
)


def command(args: list[str], *, root: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GH_HOST="github.com", GH_PROMPT_DISABLED="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, check=check, timeout=90)


def git(args: list[str], *, root: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    return command(
        ["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential", *args],
        root=root,
        check=check,
    )


def gh_json(args: list[str], *, root: Path = ROOT):
    return json.loads(command(["gh", *args], root=root).stdout)


def validate_repo_name(name: str) -> str:
    if not re.fullmatch(r"billlza/[A-Za-z0-9][A-Za-z0-9._-]{0,99}", name) or ".." in name:
        raise ValueError("Expected billlza/<repository-name>; no URL or other account is accepted")
    return name


def restore_history(root: Path) -> None:
    if (root / ".git").exists():
        return
    bundle = root / "history.bundle"
    if not bundle.is_file() or bundle.is_symlink():
        raise ValueError(
            "No Git history. Use the delivered source archive with history.bundle, or clone the Git bundle."
        )
    # Restore only metadata, without overwriting source files or inventing a new commit.
    with tempfile.TemporaryDirectory(prefix="isac-history-") as temp:
        clone = Path(temp) / "repo"
        command(["git", "clone", "--no-checkout", "--branch", "main", str(bundle), str(clone)], root=root)
        command(["git", "remote", "remove", "origin"], root=clone)
        shutil.move(str(clone / ".git"), str(root / ".git"))
    command(["git", "reset", "--mixed", "HEAD"], root=root)


def local_commit(root: Path) -> str:
    if (
        Path(command(["git", "rev-parse", "--show-toplevel"], root=root).stdout.strip()).resolve()
        != root.resolve()
    ):
        raise ValueError("Refusing to publish a parent repository")
    if command(["git", "status", "--porcelain"], root=root).stdout.strip():
        raise ValueError("Working tree is dirty; inspect and commit deliberately, then rerun full CI")
    tracked = set(command(["git", "ls-files", "-z"], root=root).stdout.split("\0"))
    if not REQUIRED_FILES.issubset(tracked):
        raise ValueError("Missing committed release files: " + ", ".join(sorted(REQUIRED_FILES - tracked)))
    sha = command(["git", "rev-parse", "HEAD"], root=root).stdout.strip()
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Invalid local commit SHA")
    return sha


def validate_repo(info: dict, target: str) -> None:
    if info.get("full_name") != target or info.get("private") is not True:
        raise ValueError(
            "Expected the exact PRIVATE repository; visibility will not be changed automatically"
        )
    if info.get("archived") or info.get("disabled"):
        raise ValueError("Repository is archived or disabled")


def validate_run(run: dict, sha: str) -> None:
    if run.get("headSha") != sha or run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("Workflow does not confirm success on the exact commit")
    jobs = run.get("jobs", [])
    if not jobs or not any(j.get("name") == "ci-gate" for j in jobs):
        raise ValueError("Missing required ci-gate job")
    if any(j.get("status") != "completed" or j.get("conclusion") != "success" for j in jobs):
        raise ValueError("A workflow job failed, was cancelled, or was skipped")
    if not str(run.get("url", "")).startswith("https://github.com/"):
        raise ValueError("Missing GitHub workflow URL")


def write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def verify_remote(target: str, sha: str, root: Path) -> None:
    commit = gh_json(["api", f"repos/{target}/commits/main"], root=root)
    if commit.get("sha") != sha:
        raise ValueError("Remote main does not equal the local tested commit")
    for name in ("uv.lock", ".python-version", ".github/workflows/tests.yml"):
        item = gh_json(["api", f"repos/{target}/contents/{name}?ref={sha}"], root=root)
        if (
            item.get("encoding") != "base64"
            or base64.b64decode(item["content"]) != (root / name).read_bytes()
        ):
            raise ValueError(f"Remote file mismatch: {name}")


def wait_ci(target: str, sha: str, receipt: dict, receipt_path: Path, timeout: int, root: Path) -> None:
    deadline = time.monotonic() + timeout
    run_id = None
    while time.monotonic() < deadline:
        if run_id is None:
            runs = gh_json(
                [
                    "run",
                    "list",
                    "--repo",
                    target,
                    "--workflow",
                    "tests.yml",
                    "--branch",
                    "main",
                    "--commit",
                    sha,
                    "--limit",
                    "20",
                    "--json",
                    "databaseId,headSha,event",
                ],
                root=root,
            )
            matches = [
                r for r in runs if r.get("headSha") == sha and r.get("event") in {"push", "workflow_dispatch"}
            ]
            if matches:
                run_id = matches[0]["databaseId"]
        if run_id is not None:
            run = gh_json(
                [
                    "run",
                    "view",
                    str(run_id),
                    "--repo",
                    target,
                    "--json",
                    "url,headSha,status,conclusion,jobs",
                ],
                root=root,
            )
            receipt.update(workflow_run_id=run_id, workflow_run_url=run.get("url"), workflow=run)
            write_receipt(receipt_path, receipt)
            if run.get("status") == "completed":
                validate_run(run, sha)
                receipt["ci_status"] = "passed"
                write_receipt(receipt_path, receipt)
                return
        time.sleep(5)
    receipt["ci_status"] = "timed_out"
    write_receipt(receipt_path, receipt)
    raise TimeoutError(
        "No successful exact-commit CI within the wait deadline; upload status remains separate"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", default="billlza/isac-world-model")
    parser.add_argument("--report", type=Path, default=ROOT / "validation/v041/ci-report.json")
    parser.add_argument(
        "--execute", action="store_true", help="Authorize a private repository creation/push and CI polling"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Read remote commit and CI only; never create/push"
    )
    parser.add_argument("--no-wait", action="store_true", help="Leave CI pending explicitly")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    target = validate_repo_name(args.repository)
    if not 1 <= args.timeout <= 7200:
        parser.error("timeout must be 1..7200 seconds")
    proof = verify(args.report.resolve(), ROOT)
    restore_history(ROOT)
    sha = local_commit(ROOT)
    origin = command(["git", "remote", "get-url", "origin"], check=False)
    if origin.returncode == 0 and origin.stdout.strip() != f"https://github.com/{target}.git":
        raise ValueError("Existing origin points elsewhere; inspect/remove it before first publication")
    receipt = {
        "schema_version": 1,
        "repository": target,
        "local_commit": sha,
        "source_tree_sha256": proof["source_tree_sha256"],
        "upload_status": "not_started",
        "ci_status": "not_started",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not args.execute:
        print(json.dumps(dict(receipt, mode="offline-preflight", status="passed"), indent=2))
        return 0
    if not shutil.which("gh"):
        raise FileNotFoundError(
            "GitHub CLI is required; install gh and log in locally. Do not send tokens to chat."
        )
    command(["gh", "auth", "status", "--hostname", "github.com"])
    if gh_json(["api", "user"]).get("login") != "billlza":
        raise ValueError("Wrong GitHub account")
    output = (
        ROOT
        / "validation/publication"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid()))
        / "receipt.json"
    )
    write_receipt(output, receipt)
    try:
        existing = command(["gh", "api", f"repos/{target}"], check=False)
        if existing.returncode:
            if args.verify_only or "HTTP 404" not in existing.stderr:
                raise RuntimeError(
                    "Cannot read repository; no create or push attempted: " + existing.stderr.strip()
                )
            # A 404 can also mean missing permission. Creation must independently succeed.
            info = gh_json(
                [
                    "api",
                    "--method",
                    "POST",
                    "user/repos",
                    "-f",
                    f"name={target.split('/')[1]}",
                    "-F",
                    "private=true",
                    "-F",
                    "auto_init=false",
                    "-f",
                    "description=From-scratch asynchronous ISAC world-model research",
                ]
            )
            receipt["repository_created"] = True
        else:
            info = json.loads(existing.stdout)
        validate_repo(info, target)
        receipt["repository_url"] = info["html_url"]
        write_receipt(output, receipt)
        url = f"https://github.com/{target}.git"
        if not args.verify_only:
            refs = git(["ls-remote", "--heads", "--tags", url]).stdout.splitlines()
            if refs and any(line.split() != [sha, "refs/heads/main"] for line in refs):
                raise ValueError(
                    "Repository already contains other history; refusing overwrite or force-push"
                )
            origin = command(["git", "remote", "get-url", "origin"], check=False)
            if origin.returncode == 0 and origin.stdout.strip() != url:
                raise ValueError("Existing origin points elsewhere; inspect it manually")
            if origin.returncode:
                command(["git", "remote", "add", "origin", url])
            if not refs:
                git(["push", "--porcelain", "origin", "HEAD:refs/heads/main"])
        verify_remote(target, sha, ROOT)
        receipt.update(upload_status="passed", remote_commit=sha, ci_status="pending")
        write_receipt(output, receipt)
        if not args.no_wait:
            wait_ci(target, sha, receipt, output, args.timeout, ROOT)
    except Exception as exc:
        receipt["error"] = type(exc).__name__ + ": " + str(exc)
        if receipt["upload_status"] != "passed":
            receipt["upload_status"] = "failed"
        elif receipt["ci_status"] != "timed_out":
            receipt["ci_status"] = "failed_or_unverified"
        write_receipt(output, receipt)
        print(json.dumps(receipt, indent=2))
        print(f"Receipt preserved: {output}")
        return 1
    print(json.dumps(receipt, indent=2))
    print(f"Receipt preserved: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Publication stopped: {error}")
        raise SystemExit(2) from error
