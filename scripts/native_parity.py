"""Differential event-protocol test against an independent integer oracle.

No NumPy or torch is needed, allowing the same probe on minimal native runners.
The Python EventQueue is separately checked against this oracle in pytest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess


class Oracle:
    def __init__(self):
        self.now = 0
        self.pending = {}
        self.seen = set()

    def apply(self, command):
        fields = command.split()
        try:
            if not fields or any(not v.isascii() or not v.isdecimal() for v in fields[1:]):
                return "ERR"
            values = list(map(int, fields[1:]))
            if any(v > 2**64 - 1 for v in values):
                return "ERR"
            if fields[0] == "P" and len(values) == 3:
                uid, cap, arrival = values
                if (arrival < cap or arrival < self.now or uid in self.seen
                        or len(self.pending) >= 4096 or len(self.seen) >= 1_000_000):
                    return "ERR"
                self.pending[uid] = arrival
                self.seen.add(uid)
                return "OK"
            if fields[0] == "R" and len(values) == 1:
                now = values[0]
                if now < self.now:
                    return "ERR"
                self.now = now
                ready = sorted((at, uid) for uid, at in self.pending.items() if at <= now)
                for _, uid in ready:
                    del self.pending[uid]
                return "READY " + ",".join(str(uid) for _, uid in ready)
        except (ValueError, OverflowError):
            return "ERR"
        return "ERR"


def make_transcript(seed=7301, count=5000):
    rng, oracle = random.Random(seed), Oracle()
    commands, expected = [], []
    for k in range(count):
        if k % 5 == 0:
            command = f"R {max(0, oracle.now + rng.randint(-2, 15))}"
        elif k % 29 == 0:
            command = "P +1 0 1"
        elif k % 31 == 0:
            command = "P 1 0 1 extra"
        else:
            uid = rng.randrange(0, max(1, k + 1))
            cap = max(0, oracle.now + rng.randint(-10, 10))
            command = f"P {uid} {cap} {max(0, cap + rng.randint(-3, 40))}"
        commands.append(command)
        expected.append(oracle.apply(command))
    commands.append(f"R {2**64 - 1}")
    expected.append(oracle.apply(commands[-1]))
    return commands, expected


def verify(binary: Path, seeds=(7301, 17, 99, 201, 599), count=5000):
    binary = binary.resolve()
    reports = []
    for seed in seeds:
        commands, expected = make_transcript(seed, count)
        result = subprocess.run([str(binary)], input="\n".join(commands) + "\n",
                                text=True, capture_output=True, check=True, timeout=30)
        actual = ["ERR" if line.startswith("ERR ") else line for line in result.stdout.splitlines()]
        if actual != expected:
            for index, pair in enumerate(zip(actual, expected)):
                if pair[0] != pair[1]:
                    raise AssertionError(f"seed={seed} command={index}: {pair}")
            raise AssertionError(f"seed={seed}: response count mismatch")
        reports.append({"seed": seed, "commands": len(commands), "status": "passed"})
    for payload in (b"P " + b"0" * 4096 + b"\n", b"\xff\n"):
        result = subprocess.run([str(binary)], input=payload, capture_output=True, timeout=5)
        if result.returncode == 0:
            raise AssertionError("Oversized / non-UTF8 protocol input did not fail")
    return {"status": "passed", "seeds": reports, "malformed_stream_tests": 2,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.binary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
