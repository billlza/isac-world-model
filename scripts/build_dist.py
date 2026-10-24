"""Build with the installed PEP-517 backend; no implicit network downloads."""

import argparse
from pathlib import Path

import setuptools.build_meta

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
print(setuptools.build_meta.build_wheel(str(args.out.resolve())))
print(setuptools.build_meta.build_sdist(str(args.out.resolve())))
