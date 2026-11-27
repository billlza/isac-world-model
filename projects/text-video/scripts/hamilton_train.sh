#!/bin/bash -l
# Invoked by submit_hamilton.sh inside one whole-H200 Slurm allocation.
set -euo pipefail
umask 077
if [[ $# != 1 || ! -f "$1" ]]; then
    echo 'Usage: hamilton_train.sh OUT.submission/request.json' >&2
    exit 2
fi
export TEXT_VIDEO_REQUEST="$1"
status_dir=$(cd -- "$(dirname -- "$1")" && pwd)
shell_error() {
    local code=$1
    trap - ERR TERM INT HUP
    printf '{"status":"error","phase":"allocation_or_module_startup","exit_code":%s}\n' "$code" > "$status_dir/job_status.json.pending"
    mv -- "$status_dir/job_status.json.pending" "$status_dir/job_status.json"
    exit "$code"
}
trap 'shell_error $?' ERR
trap 'shell_error 143' TERM
trap 'shell_error 130' INT
trap 'shell_error 129' HUP
printf '{"status":"preflight","phase":"module"}\n' > "$status_dir/job_status.json"
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ || -n ${SLURM_ARRAY_JOB_ID:-} ]]; then
    echo 'A single Slurm allocation is required; no login-node training.' >&2
    false
fi
module load python/3.13.9cuda
python_bin=/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python
test -x "$python_bin"
# Slurm executes a spool copy of this script. --chdir, not BASH_SOURCE, pins the checkout.
export TEXT_VIDEO_REPO=$(pwd -P)
export PYTHONPATH="$TEXT_VIDEO_REPO/projects/text-video/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$python_bin" - <<'PY'
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(os.environ['TEXT_VIDEO_REPO']).resolve()
REQUEST = Path(os.environ['TEXT_VIDEO_REQUEST']).resolve()
STATUS = REQUEST.parent / 'job_status.json'
child = None


def write(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def status(state, **extra):
    write(STATUS, {'status': state, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                   'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra})


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def check_sources(request):
    if str(ROOT) != request['source_root'] or git('rev-parse', 'HEAD') != request['git_commit']:
        raise RuntimeError('Source checkout no longer matches the pinned revision')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Source checkout is no longer clean')
    if any(digest(ROOT / path) != expected for path, expected in request['source_sha256'].items()):
        raise RuntimeError('Pinned source contents changed')
    if any(digest(Path(value['path'])) != value['sha256'] for value in request['inputs'].values()):
        raise RuntimeError('Pinned configuration, manifest or initialization checkpoint changed')


def check_media_dependencies():
    versions = {}
    for module_name, distribution in (('PIL.Image', 'Pillow'), ('imageio.v3', 'imageio'), ('imageio_ffmpeg', 'imageio-ffmpeg')):
        try:
            package = importlib.import_module(module_name.split('.')[0])
            importlib.import_module(module_name)
            version = getattr(package, '__version__', None)
            if not version:
                raise RuntimeError('package did not expose its version')
        except Exception as exc:
            raise RuntimeError(f'Required media dependency {distribution} cannot be imported and versioned by '
                               f'{sys.executable} with PYTHONNOUSERSITE=1: {exc}. '
                               'No dependencies were installed or runtime settings changed.') from exc
        versions[distribution] = str(version)
    return versions


def interrupted(signum, frame):
    if child is not None and child.poll() is None:
        child.terminate()
    raise RuntimeError(f'Allocation interrupted by signal {signum}')


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, interrupted)
try:
    status('preflight', phase='source_and_runtime')
    request = json.loads(REQUEST.read_text())
    content = {key: value for key, value in request.items() if key != 'identity'}
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != request['identity']:
        raise RuntimeError('Submission request was modified after preparation')
    if request['stage'] not in ('pretrain', 'sft') or ((request['stage'] == 'sft') != ('init_checkpoint' in request['inputs'])):
        raise RuntimeError('Invalid stage or initialization checkpoint contract')
    if not re.fullmatch(r'[0-9]+', os.environ.get('SLURM_JOB_ID', '')) or os.environ.get('SLURM_ARRAY_JOB_ID'):
        raise RuntimeError('A single Slurm allocation is required; no login-node training')
    check_sources(request)
    out = Path(request['out'])
    if out.exists():
        raise RuntimeError('OUT already exists; refusing to overwrite training artifacts')
    expected_python = '/apps/developers/compilers/python/3.13.9cuda/1/default/bin/python'
    if Path(sys.executable).resolve() != Path(expected_python).resolve():
        raise RuntimeError('Unexpected Python executable')
    import numpy as np
    import torch
    observed = ['.'.join(map(str, sys.version_info[:3])), np.__version__, str(torch.__version__), torch.version.cuda]
    expected = ['3.13.9', '2.3.4', '2.9.0+cu128', '12.8']
    if observed != expected:
        raise RuntimeError(f'School runtime changed: {observed!r}; expected {expected!r}')
    media_versions = check_media_dependencies()
    runtime = {'python_executable': sys.executable, 'observed': observed, 'expected': expected,
               'media_dependencies': media_versions, 'cuda_forward_backward': 'not_executed'}
    write(REQUEST.parent / 'runtime.json', runtime)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA device is required; CPU fallback is prohibited')
    gpu = torch.cuda.get_device_properties(0)
    if 'H200' not in gpu.name or 'MIG' in gpu.name or gpu.total_memory < 120 * 1024**3:
        raise RuntimeError('Expected a whole H200; refusing a different or fractional GPU allocation')
    left = torch.randn(256, 256, device='cuda', requires_grad=True)
    right = torch.randn(256, 256, device='cuda', requires_grad=True)
    loss = (left @ right).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item() or any(t.grad is None or not torch.isfinite(t.grad).all().item() for t in (left, right)):
        raise RuntimeError('CUDA forward/backward preflight failed')
    del left, right, loss
    torch.cuda.empty_cache()
    inputs = request['inputs']
    validation = subprocess.run([sys.executable, '-m', 'text_video', 'validate-data', '--manifest', inputs['manifest']['path'],
                                 '--config', inputs['config']['path']], cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(validation.stdout)
    if report['data_fingerprint'] != request['data_validation']['data_fingerprint']:
        raise RuntimeError('Dataset assets changed since submission preparation')
    command = [sys.executable, '-m', 'text_video', 'train', '--config', inputs['config']['path'],
               '--manifest', inputs['manifest']['path'], '--out', str(out), '--device', 'cuda', '--stage', request['stage']]
    if request['stage'] == 'sft':
        command += ['--init-checkpoint', inputs['init_checkpoint']['path']]
    runtime.update({'gpu': gpu.name, 'gpu_memory_bytes': gpu.total_memory, 'cuda_forward_backward': 'passed',
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'command': command,
          'data_fingerprint': report['data_fingerprint'], 'git_commit': request['git_commit'],
          'RL': 'not_implemented_not_executed', 'RLHF': 'not_implemented_not_executed'})
    write(REQUEST.parent / 'runtime.json', runtime)
    check_sources(request)
    status('running', stage=request['stage'])
    child = subprocess.Popen(command, cwd=ROOT)
    if child.wait() != 0:
        raise RuntimeError(f'Training exited with code {child.returncode}')
    app_status = json.loads((out / 'status.json').read_text())
    if app_status.get('status') != 'completed':
        raise RuntimeError('Training did not report completed status')
    check_sources(request)
    status('completed', stage=request['stage'], app_status=app_status)
except BaseException as exc:
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    status('error', error=repr(exc))
    raise
PY
