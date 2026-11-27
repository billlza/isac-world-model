#!/usr/bin/env bash
# Local Hamilton launcher; never connects by SSH. Default: validate and prepare only.
# Usage: submit_hamilton.sh [--check|--submit] EXACT_SHA CONFIG MANIFEST OUT [pretrain|sft] [INIT_CHECKPOINT]
# OUT must not exist. Preparation and the permanent submission claim live in OUT.submission.
set -euo pipefail
umask 077
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TEXT_VIDEO_REPO=$(cd -- "$script_dir/../../.." && pwd)
export PYTHONPATH="$TEXT_VIDEO_REPO/projects/text-video/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "${TEXT_VIDEO_LAUNCH_PYTHON:-python3}" - "$@" <<'PY'
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys

ROOT = Path(os.environ['TEXT_VIDEO_REPO']).resolve()


def fail(message):
    raise SystemExit(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


parser = argparse.ArgumentParser(description='Prepare by default; --submit explicitly submits once from a clean Hamilton checkout.')
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--check', action='store_true', help='Validate data and prepare the exact sbatch command; never submit (default).')
mode.add_argument('--submit', action='store_true', help='Submit once on a Hamilton login host; preserve ambiguous claims for manual inspection.')
parser.add_argument('exact_sha', help='Full 40-character commit SHA; HEAD and the clean worktree must match.')
parser.add_argument('config', type=Path)
parser.add_argument('manifest', type=Path)
parser.add_argument('out', type=Path)
parser.add_argument('stage', choices=('pretrain', 'sft'), nargs='?', default='pretrain')
parser.add_argument('init_checkpoint', type=Path, nargs='?')
args = parser.parse_args()
if not re.fullmatch(r'[0-9a-f]{40}', args.exact_sha) or git('rev-parse', 'HEAD') != args.exact_sha:
    fail('EXACT_SHA must be the full current HEAD commit; refusing an unpinned checkout.')
if git('status', '--porcelain', '--untracked-files=all'):
    fail('A clean committed checkout is required, including untracked source files.')
if (args.stage == 'sft') != (args.init_checkpoint is not None):
    fail('sft requires INIT_CHECKPOINT; pretrain must start randomly and rejects INIT_CHECKPOINT.')

out = args.out.resolve()
prepared = out.with_name(out.name + '.submission')
if out.exists():
    fail('OUT already exists; preserve it and choose a new output directory.')
if prepared.is_relative_to(ROOT):
    ignored = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', str(prepared)], check=False)
    if ignored.returncode:
        fail('Use an output outside the checkout or under an ignored directory such as runs/.')
inputs = {}
for name in ('config', 'manifest', 'init_checkpoint'):
    value = getattr(args, name)
    if value is not None:
        path = value.resolve()
        if not path.is_file():
            fail(f'{name} must be an existing file: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest(path)}
source_paths = git('ls-files', '--', 'projects/text-video').splitlines()
required = {'projects/text-video/scripts/submit_hamilton.sh', 'projects/text-video/scripts/hamilton_train.sh',
            'projects/text-video/src/text_video/__main__.py'}
if not required.issubset(source_paths):
    fail('All launcher and application source files must be committed.')
source_hashes = {path: digest(ROOT / path) for path in source_paths}


def check_pins():
    if git('rev-parse', 'HEAD') != args.exact_sha or git('status', '--porcelain', '--untracked-files=all'):
        fail('The pinned checkout changed during preparation; refusing submission.')
    if any(digest(ROOT / path) != expected for path, expected in source_hashes.items()):
        fail('Pinned source contents changed during preparation.')
    if any(digest(Path(value['path'])) != value['sha256'] for value in inputs.values()):
        fail('Pinned configuration, manifest or initialization checkpoint changed during preparation.')


validation_command = [sys.executable, '-m', 'text_video', 'validate-data', '--manifest', inputs['manifest']['path'], '--config', inputs['config']['path']]
validation = subprocess.run(validation_command, cwd=ROOT, check=False, capture_output=True, text=True)
if validation.returncode:
    fail('Dataset validation failed:\n' + validation.stderr + validation.stdout)
try:
    report = json.loads(validation.stdout)
    if not re.fullmatch(r'[0-9a-f]{64}', report['data_fingerprint']):
        raise ValueError('invalid fingerprint')
except (ValueError, KeyError, TypeError) as exc:
    fail(f'validate-data did not return a stable data_fingerprint: {exc}')
check_pins()
request = {'schema_version': 1, 'source_root': str(ROOT), 'git_commit': args.exact_sha,
           'source_sha256': source_hashes, 'inputs': inputs, 'out': str(out), 'stage': args.stage,
           'data_validation': report, 'initialization': 'random' if args.stage == 'pretrain' else 'checkpoint',
           'RL': 'not_implemented_not_executed', 'RLHF': 'not_implemented_not_executed'}
identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
request['identity'] = identity
manifest_path = prepared / 'request.json'
command = ['sbatch', '--parsable', '-p', 'cuda', '--gres=gpu:h200_nvl:1', '--time=04:00:00',
           '--job-name=text-video-' + args.stage, '--output=' + str(prepared / 'slurm-%j.out'),
           '--error=' + str(prepared / 'slurm-%j.err'), '--chdir=' + str(ROOT),
           str(ROOT / 'projects/text-video/scripts/hamilton_train.sh'), str(manifest_path)]
try:
    prepared.mkdir(parents=True)
except FileExistsError:
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != request:
        fail('Existing preparation differs or is incomplete; preserve it and use a new OUT.')
else:
    write(manifest_path, request)
    write(prepared / 'command.json', {'command': command, 'shell': shlex.join(command), 'created_utc': now()})

claim = prepared / 'claim'
if claim.exists():
    receipt_path = claim / 'receipt'
    receipt = receipt_path.read_text().strip() if receipt_path.is_file() else ''
    if re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', receipt):
        print(json.dumps({'status': 'already_submitted', 'slurm_job_id': receipt.split(';')[0], 'receipt': str(receipt_path)}))
        raise SystemExit(0)
    fail('Submission claim exists without a confirmed receipt. Inspect squeue --me, receipt.pending and submission.stderr; do not retry blindly.')
if not args.submit:
    print(json.dumps({'status': 'prepared_not_submitted', 'request': str(manifest_path), 'sbatch': command,
                      'gpu_validation': 'not_executed', 'next': 'Repeat the same invocation with --submit on Hamilton.'}))
    raise SystemExit(0)
if socket.getfqdn() not in ('login1.ham8.dur.ac.uk', 'login2.ham8.dur.ac.uk'):
    fail('Actual submission requires a Hamilton login host; no SSH was attempted.')
check_pins()
try:
    claim.mkdir()
except FileExistsError:
    fail('Another process claimed this submission; inspect the existing claim before any further action.')
write(claim / 'status.json', {'status': 'submitting', 'identity': identity, 'updated_utc': now()})
environment = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
try:
    with (claim / 'receipt.pending').open('x') as stdout, (claim / 'submission.stderr').open('x') as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False)
    receipt = (claim / 'receipt.pending').read_text().strip()
    if result.returncode or not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', receipt):
        raise RuntimeError('No confirmed scheduler receipt')
except BaseException as exc:
    write(claim / 'status.json', {'status': 'submission_unconfirmed', 'error': repr(exc), 'updated_utc': now()})
    fail('Submission outcome is unconfirmed; claim retained. Inspect squeue --me and submission.stderr before taking action.')
(claim / 'receipt.pending').replace(claim / 'receipt')
write(claim / 'status.json', {'status': 'submitted', 'slurm_job_id': receipt.split(';')[0], 'updated_utc': now()})
print(json.dumps({'status': 'submitted_not_completed', 'slurm_job_id': receipt.split(';')[0],
                  'receipt': str(claim / 'receipt'), 'out': str(out)}))
PY
