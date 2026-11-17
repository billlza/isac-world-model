"""Verify pinned source and completed pretraining evidence, including model weights.

The archive records the job's status at packaging time. A separate terminal
status.json and the scheduler receipt remain authoritative for job completion.
This module uses only the standard library and never deserializes checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHOOL_RUNTIME = {"python": "3.13.9", "torch": "2.9.0+cu128", "numpy": "2.3.4", "device": "cuda"}
MAX_JSON_BYTES = 64 * 1024**2
MAX_FILE_BYTES = 2 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular_root(path):
    root = Path(path).absolute()
    require(root.is_dir() and not root.is_symlink(), f"Missing or non-regular directory: {root}")
    return root


def regular_file(root, relative):
    require(isinstance(relative, str), "Artifact path must be a string")
    parts = PurePosixPath(relative).parts
    require(
        bool(parts)
        and not relative.startswith("/")
        and ".." not in parts
        and "\\" not in relative
        and "\x00" not in relative
        and PurePosixPath(relative).as_posix() == relative,
        f"Invalid relative artifact path: {relative!r}",
    )
    path = root
    for component in parts:
        path = path / component
        require(not path.is_symlink(), f"Symlink artifact is not allowed: {relative}")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), f"Missing or non-regular file: {relative}")
    require(path.stat().st_size <= MAX_FILE_BYTES, f"Artifact exceeds size limit: {relative}")
    return path


def digest_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def metadata(path):
    return {"sha256": digest_file(path), "bytes": path.stat().st_size}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    require(path.stat().st_size <= MAX_JSON_BYTES, f"JSON exceeds size limit: {path.name}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
    # JSON's 1e1000 parses to infinity without invoking parse_constant.
    json.dumps(value, allow_nan=False)
    return value


def write_new(path, raw):
    require(not path.exists() and not path.is_symlink(), f"Output already exists: {path}")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def source_snapshot(source_root, pin):
    """Tie each current source byte to both its committed Git blob and SHA256."""
    root = regular_root(source_root)
    require(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{40}", pin), "Invalid source pin")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    require(git("rev-parse", "HEAD").decode().strip() == pin, "Source HEAD differs from pin")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    require(
        not git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "*.py").strip(),
        "Untracked source could shadow the pinned implementation",
    )
    files = {}
    entries = git("ls-tree", "-r", "-z", "--full-tree", pin).split(b"\0")
    for entry in entries:
        if not entry:
            continue
        info, filename = entry.split(b"\t", 1)
        mode, kind, blob = info.decode("ascii").split()
        relative = filename.decode("utf-8")
        require(kind == "blob" and mode in {"100644", "100755"}, f"Unsupported tracked source: {relative}")
        path = regular_file(root, relative)
        git_hash = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                git_hash.update(block)
                sha256.update(block)
        require(git_hash.hexdigest() == blob, f"Source bytes differ from pinned Git blob: {relative}")
        files[relative] = {"sha256": sha256.hexdigest(), "bytes": path.stat().st_size, "git_blob": blob}
    require(files, "Pinned source tree is empty")
    return {"schema_version": 1, "git_commit": pin, "files": files}


def record_source(source_root, pin, destination):
    snapshot = source_snapshot(source_root, pin)
    write_new(Path(destination), json_bytes(snapshot))
    return snapshot


def study_files(root):
    """Reject symlinks anywhere in the result tree, including directory links."""
    study_root = regular_root(root / "study")
    result = {}
    for directory, directories, filenames in os.walk(study_root, followlinks=False):
        for name in directories:
            require(not (Path(directory) / name).is_symlink(), f"Symlink result directory: {name}")
        for name in filenames:
            path = Path(directory) / name
            relative = path.relative_to(study_root).as_posix()
            result[relative] = regular_file(study_root, relative)
    return result


def verify_evidence(run_dir, source_root, expected_profile=None):
    root = regular_root(run_dir)
    results = study_files(root)
    require("study.json" in results, "Missing study.json")
    study = read_json(results["study.json"])
    profile = study.get("profile")
    require(
        study.get("schema_version") == 1
        and study.get("status") == "completed"
        and profile in {"formal", "development", "smoke"}
        and (expected_profile is None or profile == expected_profile),
        "Expected completed matching pretraining study profile",
    )
    base = {"source_manifest.json": regular_file(root, "source_manifest.json")}
    recorded_source = read_json(base["source_manifest.json"])
    for name in ("preflight.json", "status.json"):
        if profile == "formal" or (root / name).exists() or (root / name).is_symlink():
            base[name] = regular_file(root, name)
    preflight = read_json(base["preflight.json"]) if "preflight.json" in base else None
    status = read_json(base["status.json"]) if "status.json" in base else None
    revision, job = study.get("git_commit", ""), study.get("slurm_job_id", "")
    require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        and (
            (isinstance(job, str) and re.fullmatch(r"[0-9]+", job)) or (profile != "formal" and job is None)
        ),
        "Invalid pinned source or Slurm identity",
    )
    require(isinstance(study.get("runtime"), dict), "Missing study runtime")
    if profile == "formal":
        require(
            all(study["runtime"].get(key) == value for key, value in SCHOOL_RUNTIME.items()),
            "Pinned Hamilton study runtime differs",
        )
        preflight_runtime = preflight.get("runtime", {})
        require(
            preflight_runtime.get("python") == SCHOOL_RUNTIME["python"]
            and preflight_runtime.get("torch") == SCHOOL_RUNTIME["torch"]
            and preflight_runtime.get("torch_cuda") == "12.8",
            "Pinned Hamilton preflight runtime differs",
        )
        gpu = preflight.get("gpu", {})
        require(
            "H200" in gpu.get("name", "")
            and gpu.get("mig_mode") == "Disabled"
            and gpu.get("total_memory_bytes", 0) >= 120 * 1024**3,
            "Whole H200 evidence is missing",
        )
    if preflight is not None:
        require(
            preflight.get("status") == "passed"
            and preflight.get("git_commit") == revision
            and preflight.get("slurm", {}).get("job_id") == job,
            "Preflight and study identities differ",
        )
    if status is not None:
        require(
            status.get("git_commit") == revision and status.get("slurm_job_id") == job,
            "Study and job identities differ",
        )
        require(
            (status.get("status"), status.get("stage")) in {("running", "package"), ("completed", "done")}
            and (
                (status["status"] == "running" and status.get("exit_code") is None)
                or (status["status"] == "completed" and status.get("exit_code") == 0)
            ),
            "Job has not reached successful study packaging",
        )
    require(
        recorded_source == source_snapshot(source_root, revision), "Source manifest changed since job start"
    )
    recorded = study.get("artifacts_sha256")
    require(
        isinstance(recorded, dict) and set(recorded) == set(results) - {"study.json"},
        "Study artifact digest inventory is incomplete or contains unexpected paths",
    )
    for relative, path in results.items():
        if relative != "study.json":
            require(digest_file(path) == recorded[relative], f"Changed result evidence: {relative}")
        if path.suffix == ".json":
            read_json(path)
    cells = study.get("cells")
    require(isinstance(cells, dict) and bool(cells), "Missing completed training cells")
    planned = study.get("planned_cells")
    require(
        isinstance(planned, list)
        and len(planned) == len(cells)
        and all(isinstance(name, str) for name in planned)
        and set(planned) == set(cells),
        "Completed cells differ from the planned training cells",
    )
    if "cell_order" in study:
        require(study["cell_order"] == planned, "Study cell order differs from planned cells")
    checkpoints = {}
    for name, cell in cells.items():
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", name)
            and isinstance(cell, dict)
            and cell.get("status") == "completed"
            and cell.get("path") == name,
            "Invalid or incomplete training cell",
        )
        prefix = f"{name}/"
        require(
            cell.get("artifacts_sha256")
            == {key: value for key, value in recorded.items() if key.startswith(prefix)},
            f"Cell artifact inventory differs: {name}",
        )
        for evidence in ("manifest.json", "history.json", "metrics.json"):
            require(prefix + evidence in results, f"Missing cell evidence: {prefix + evidence}")
            require(
                cell.get(evidence.removesuffix(".json")) == prefix + evidence,
                f"Cell evidence path differs: {name}",
            )
        for filename in ("model.pt", "final.pt", "resume.pt"):
            relative = prefix + filename
            require(
                relative in results and relative + ".sha256" in results,
                f"Missing checkpoint or digest: {relative}",
            )
            expected = results[relative + ".sha256"].read_text(encoding="ascii").strip()
            require(re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid checkpoint digest: {relative}")
            require(
                results[relative].stat().st_size > 0 and digest_file(results[relative]) == expected,
                f"Checkpoint checksum differs: {relative}",
            )
            checkpoints[relative] = {**metadata(results[relative]), "weights_included": True}
    expected_weights = set(checkpoints)
    require(
        {name for name in results if name.endswith(".pt")} == expected_weights,
        "Unexpected checkpoint outside registered cells",
    )
    files = {"study/" + name: path for name, path in results.items()}
    for name, path in base.items():
        files["job_status_at_packaging.json" if name == "status.json" else name] = path
    return study, files, {"study/" + name: value for name, value in checkpoints.items()}


def package(run_dir, destination, source_root, expected_profile=None):
    out = Path(destination).absolute()
    staging = out.with_suffix(out.suffix + ".writing")
    checksum_path = out.with_suffix(out.suffix + ".sha256")
    for path in (out, staging, checksum_path):
        require(
            not path.exists() and not path.is_symlink(), f"Evidence archive output already exists: {path}"
        )
    study, files, checkpoints = verify_evidence(run_dir, source_root, expected_profile)
    artifacts = {name: metadata(path) for name, path in sorted(files.items())}
    manifest = {
        "schema_version": 1,
        "kind": "isac-complex-scene-pretraining",
        "status": "verified_training_evidence",
        "completion_scope": (
            "Training evidence verified; terminal scheduler and job status are supplied separately"
            if study["profile"] == "formal"
            else f"{study['profile'].capitalize()} training evidence verified; no formal scheduler submission is claimed"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": study["git_commit"],
        "slurm_job_id": study["slurm_job_id"],
        "profile": study["profile"],
        "weights_included": True,
        "checkpoints": checkpoints,
        "artifacts": artifacts,
    }
    payload = json_bytes(manifest)
    # Stream checkpoint files into the archive; do not retain every weight in RAM.
    with staging.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for name, path in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode = artifacts[name]["bytes"], 0o600
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            info = tarfile.TarInfo("bundle_manifest.json")
            info.size, info.mode = len(payload), 0o600
            archive.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    # Verify the actual tar payload, not only the inputs from before packaging.
    with tarfile.open(staging, mode="r:gz") as archive:
        require(set(archive.getnames()) == set(files) | {"bundle_manifest.json"}, "Archive inventory differs")
        for name, expected in artifacts.items():
            member = archive.getmember(name)
            require(member.isfile() and member.size == expected["bytes"], f"Archive size differs: {name}")
            checksum = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            require(checksum.hexdigest() == expected["sha256"], f"Archive checksum differs: {name}")
    # Recheck that result/source files did not change while the tar was written.
    verify_evidence(run_dir, source_root, expected_profile)
    require(
        all(metadata(files[name]) == expected for name, expected in artifacts.items()),
        "Evidence changed during packaging",
    )
    archive_sha256 = digest_file(staging)
    os.link(staging, out)  # Exclusive publication; never replace a previous archive.
    staging.unlink()
    write_new(checksum_path, f"{archive_sha256}  {out.name}\n".encode())
    return {
        "status": "packaged",
        "archive": str(out),
        "sha256": archive_sha256,
        "bytes": out.stat().st_size,
        "weights_included": True,
        "artifacts_verified": len(files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source", help="Record the initial pinned source digest inventory")
    source.add_argument("--source-root", required=True, type=Path)
    source.add_argument("--pin", required=True)
    source.add_argument("--out", required=True, type=Path)
    pack = commands.add_parser("package", help="Verify and archive all results and model weights")
    pack.add_argument("--source-root", required=True, type=Path)
    pack.add_argument("--run-dir", required=True, type=Path)
    pack.add_argument("--out", required=True, type=Path)
    pack.add_argument("--expected-profile", choices=("formal", "development", "smoke"))
    args = parser.parse_args()
    if args.command == "source":
        snapshot = record_source(args.source_root, args.pin, args.out)
        print(
            json.dumps(
                {
                    "status": "source_verified",
                    "git_commit": snapshot["git_commit"],
                    "files": len(snapshot["files"]),
                }
            )
        )
    else:
        print(
            json.dumps(package(args.run_dir, args.out, args.source_root, args.expected_profile)), flush=True
        )


if __name__ == "__main__":
    main()
