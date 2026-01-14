#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path


def run_tests(coverage=False, verbose=False, specific_file=None):
    cmd = ['pytest']

    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')

    if coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing'])

    if specific_file:
        cmd.append(f'tests/{specific_file}')
    else:
        cmd.append('tests/')

    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    return result.returncode


def main():
    import argparse

    parser = argparse.ArgumentParser(description='运行测试')
    parser.add_argument('--coverage', '-c', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--file', '-f', type=str)

    args = parser.parse_args()

    returncode = run_tests(
        coverage=args.coverage,
        verbose=args.verbose,
        specific_file=args.file
    )

    sys.exit(returncode)


if __name__ == '__main__':
    main()
