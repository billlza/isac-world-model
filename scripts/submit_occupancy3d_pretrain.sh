#!/usr/bin/env bash
# Prepare by default. Run --submit only in the published, clean Hamilton checkout.
# No SSH, account changes, dependency installation, or automatic submission.
set -euo pipefail
umask 077
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ISAC_OCCUPANCY_SCRIPT_ROOT="$script_dir/.."
exec "${ISAC_LAUNCH_PYTHON:-python3}" - "$@" <<'PY'
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

ROOT = Path(os.environ['ISAC_OCCUPANCY_SCRIPT_ROOT']).resolve()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def hashes(paths):
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


parser = argparse.ArgumentParser(description='Prepare or explicitly submit the registered 3D H200 pretraining job.')
parser.add_argument('--config', type=Path, default=ROOT / 'configs/occupancy3d_h200_pretrain.json')
parser.add_argument('--output-root', type=Path, help='Directory for prepared manifest, receipt and eventual study; default runs/<source-config fingerprint>.')
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--submit', action='store_true', help='Submit once, on Hamilton login host only.')
mode.add_argument('--run-job', type=Path, help=argparse.SUPPRESS)
args = parser.parse_args()

if args.run_job:
    manifest_path = args.run_job.resolve()
    manifest = json.loads(manifest_path.read_text())
    config = manifest['configuration']
    output = manifest_path.parent
    job_id = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'[0-9]+', job_id) or os.environ.get('SLURM_ARRAY_JOB_ID'):
        raise SystemExit('A single Slurm allocation is required; no login-node training.')
    write(output / 'job_status.json', {'status': 'preflight', 'slurm_job_id': job_id, 'updated_utc': now()})
    try:
        if str(ROOT) != manifest['source_root'] or git('rev-parse', 'HEAD') != manifest['git_commit']:
            raise RuntimeError('Source checkout differs from submitted revision')
        if hashes(manifest['source_sha256']) != manifest['source_sha256']:
            raise RuntimeError('Submitted source or configuration has changed')
        import numpy as np
        import torch
        runtime = config['runtime']
        observed = ['.'.join(map(str, sys.version_info[:3])), np.__version__, str(torch.__version__), torch.version.cuda]
        expected = [runtime[k] for k in ('expected_python', 'expected_numpy', 'expected_torch', 'expected_cuda')]
        if observed != expected:
            raise RuntimeError(f'School runtime changed: {observed!r}; expected {expected!r}')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA device is required; CPU fallback prohibited')
        device = torch.cuda.get_device_properties(0)
        if 'H200' not in device.name or device.total_memory < 120 * 1024**3:
            raise RuntimeError('Requested whole H200 was not observed; refuse smaller or different allocation')
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(ROOT / 'src'), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
        command = [sys.executable, str(ROOT / config['runner']), '--output', str(output / 'study'), '--device', 'cuda', '--determinism', runtime['determinism'], '--threads', str(runtime['threads'])]
        for key, value in config['training'].items():
            command += ['--' + key.replace('_', '-')]
            command += [str(item) for item in value] if isinstance(value, list) else [str(value)]
        write(output / 'runtime.json', {'observed': observed, 'expected': expected, 'gpu': device.name, 'gpu_memory_bytes': device.total_memory, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'slurm_job_id': job_id, 'command': command, 'stages': config['stages'], 'determinism': runtime['determinism'], 'determinism_note': runtime['determinism_note'], 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
        write(output / 'job_status.json', {'status': 'training', 'slurm_job_id': job_id, 'updated_utc': now()})
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        status = json.loads((output / 'study/status.json').read_text())
        if status.get('status') != 'completed' or hashes(manifest['source_sha256']) != manifest['source_sha256']:
            raise RuntimeError('Study did not complete or submitted sources changed')
        write(output / 'job_status.json', {'status': 'completed', 'slurm_job_id': job_id, 'updated_utc': now(), 'study_status': status})
    except BaseException as exc:
        write(output / 'job_status.json', {'status': 'failed', 'slurm_job_id': job_id, 'updated_utc': now(), 'error': repr(exc)})
        raise
    raise SystemExit(0)

config_path = args.config.resolve()
config = json.loads(config_path.read_text())
if config.get('submit_default') is not False or config['runtime']['determinism'] not in ('strict', 'warn'):
    raise SystemExit('Invalid preparation default or explicit determinism policy')
paths = [str(config_path.relative_to(ROOT)), 'scripts/submit_occupancy3d_pretrain.sh', config['runner'], config['scientific_reference_runner']]
if config['training'].get('exclude_scene_hashes'):
    paths.append(config['training']['exclude_scene_hashes'])
paths += sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'src/isac_wm').rglob('*.py'))
source_hashes = hashes(paths)
identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
output = (args.output_root or ROOT / 'runs' / ('occupancy3d-h200-' + identity[:12])).resolve()
manifest_path = output / 'submission_manifest.json'
if output.exists() and not manifest_path.exists():
    raise SystemExit('Existing output directory has no valid preparation manifest; preserve and inspect it')
output.mkdir(parents=True, exist_ok=True)
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if manifest['source_sha256'] != source_hashes or manifest['source_root'] != str(ROOT):
        raise SystemExit('Preparation belongs to different source/configuration; use a new output directory')
else:
    manifest = {'schema_version': 1, 'status': 'prepared_not_submitted', 'created_utc': now(), 'source_root': str(ROOT), 'git_commit': git('rev-parse', 'HEAD'), 'source_sha256': source_hashes, 'configuration': config, 'submission_identity': identity, 'gpu_validation': 'not_executed', 'source_worktree_status': git('status', '--porcelain', '--untracked-files=normal')}
    write(manifest_path, manifest)

runtime = config['runtime']
batch = output / 'job.sh'
batch_text = '#!/bin/bash -l\nset -euo pipefail\numask 077\n'
batch_text += 'module load ' + shlex.quote(runtime['module']) + '\n'
batch_text += 'export ISAC_LAUNCH_PYTHON=' + shlex.quote(runtime['python']) + '\n'
batch_text += 'export CUBLAS_WORKSPACE_CONFIG=' + shlex.quote(runtime['CUBLAS_WORKSPACE_CONFIG']) + '\n'
batch_text += 'export OMP_NUM_THREADS={0} MKL_NUM_THREADS={0} OPENBLAS_NUM_THREADS={0} PYTHONNOUSERSITE=1\n'.format(runtime['threads'])
batch_text += 'exec bash ' + shlex.quote(str(ROOT / 'scripts/submit_occupancy3d_pretrain.sh')) + ' --run-job ' + shlex.quote(str(manifest_path)) + '\n'
if batch.exists() and batch.read_text() != batch_text:
    raise SystemExit('Prepared batch file differs; preserve it and inspect before submitting')
batch.write_text(batch_text)
if not args.submit:
    print(json.dumps({'status': 'prepared_not_submitted', 'manifest': str(manifest_path), 'batch_script': str(batch), 'gpu_validation': 'not_executed', 'next': 'On the clean published Hamilton checkout: bash scripts/submit_occupancy3d_pretrain.sh --submit'}))
    raise SystemExit(0)

if socket.getfqdn() not in ('login1.ham8.dur.ac.uk', 'login2.ham8.dur.ac.uk'):
    raise SystemExit('Submission requires a Hamilton login host; no SSH attempted')
if git('rev-parse', 'HEAD') != manifest['git_commit'] or git('diff', '--name-only', 'HEAD', '--', *paths):
    raise SystemExit('Publish and use a clean pinned source checkout before submission')
tracked = set(git('ls-files').splitlines())
if any(path not in tracked for path in paths):
    raise SystemExit('All submitted source/configuration files must be committed before submission')
claim = output / '.submission'
try:
    claim.mkdir()
except FileExistsError:
    receipt = claim / 'receipt'
    if receipt.exists() and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', receipt.read_text().strip()):
        print('Already submitted; no duplicate job. Receipt: ' + receipt.read_text().strip())
        raise SystemExit(0)
    raise SystemExit('Submission claim exists without confirmed receipt. Inspect squeue --me, receipt.pending and submission.stderr; do not blindly retry')
write(claim / 'request.json', {'manifest': str(manifest_path), 'identity': identity, 'created_utc': now()})
(claim / 'stage').write_text('submitting\n')
slurm = config['slurm']
command = ['sbatch', '--parsable', '--export=ALL', '--partition=' + slurm['partition'], '--gres=' + slurm['gres'], '--time=' + slurm['time'], '--job-name=' + slurm['job_name'], '--output=' + str(output / 'slurm-%j.out'), '--chdir=' + str(ROOT)]
if slurm['account'] is not None:
    raise SystemExit('This registered profile does not define a verified Slurm accounting project')
command.append(str(batch))
environment = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
with (claim / 'receipt.pending').open('w') as out, (claim / 'submission.stderr').open('w') as err:
    result = subprocess.run(command, env=environment, stdout=out, stderr=err, check=False)
receipt = (claim / 'receipt.pending').read_text().strip()
if result.returncode or not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', receipt):
    raise SystemExit('No confirmed scheduler receipt; claim retained. Inspect squeue --me and submission.stderr')
(claim / 'receipt.pending').replace(claim / 'receipt')
(claim / 'job-id').write_text(receipt.split(';')[0] + '\n')
(claim / 'stage').write_text('submitted\n')
print(json.dumps({'status': 'submitted_not_completed', 'slurm_job_id': receipt.split(';')[0], 'receipt': str(claim / 'receipt'), 'results': str(output)}))
PY
