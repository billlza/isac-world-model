"""Package completed metric records from one real follow-up job, without model weights."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


def package(root, out):
    names = ["preflight.json", "prior-inputs.json", "frozen.json", "crossed/crossed.json"]
    names += ["crossed/replication_diagnostic.json", "crossed/fresh_test.json"]
    names += [
        f"crossed/init-{i}-block-{b}/{f}"
        for i in (7, 11)
        for b in (7, 11)
        for f in ("manifest.json", "history.json")
    ]
    files = {name: (root / name).read_bytes() for name in names}
    f, p, x = (json.loads(files[name]) for name in ("frozen.json", "preflight.json", "crossed/crossed.json"))
    if not (
        p["status"] == "passed"
        and f["status"] == x["status"] == "completed"
        and f["analysis_commit"] == p["git_commit"] == x["git_commit"]
        and x["profile"] == "formal"
    ):
        raise ValueError("Only completed, matching formal experiments can be packaged")
    old = json.loads(files["prior-inputs.json"])
    if old["status"] != "verified" or old["job"] != f["original_array_id"]:
        raise ValueError("Source run mismatch")
    if len(f["models"]) != 12 or any(
        m["status"] != "completed" or len(m["episodes"]) != 48 for m in f["models"]
    ):
        raise ValueError("Incomplete frozen grid")
    if len(x["cells"]) != 4 or any(m["status"] != "completed" for m in x["cells"].values()):
        raise ValueError("Incomplete crossed grid")
    readout = {
        "git_commit": x["git_commit"],
        "job_id": p["slurm"]["job_id"],
        "source_array": old["job"],
        "frozen_A96": {
            str(m["training_seed"]): m["summary"]
            for m in f["models"]
            if m["architecture"] == "structured-attention" and m["width"] == 96
        },
        "fresh_crossed": x["results"]["fresh_test"],
    }
    files["readout.json"] = json.dumps(readout, indent=2, allow_nan=False).encode()
    # No completed-looking destination is exposed if validation/packing is interrupted.
    staging = out.with_suffix(out.suffix + ".writing")
    with staging.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as tar:
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    out.hardlink_to(staging)
    staging.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.run_dir, args.out)


if __name__ == "__main__":
    main()
