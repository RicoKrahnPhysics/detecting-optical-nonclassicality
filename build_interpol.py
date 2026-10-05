"""Build a within-run interpolation train/test pair from the raw TXT files."""

import argparse
from pathlib import Path

from raw_dataset import build


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw TXT -> interpolation dataset')
    parser.add_argument('--raw-dir', type=Path, required=True,
                        help='Directory containing weak_pnr_100000_*.txt for one run')
    parser.add_argument('--run', type=int, required=True, help='Run number (1-8)')
    parser.add_argument('--gap', type=int, help='One gap from 1-9; omit to build all nine')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'datasets')
    args = parser.parse_args()
    if not 1 <= args.run <= 8 or (args.gap is not None and not 1 <= args.gap <= 9):
        parser.error('run must be 1-8 and gap must be 1-9')
    gaps = [args.gap] if args.gap is not None else range(1, 10)
    build(args.raw_dir, args.out_dir, f'Run{args.run}', 'interpol', gaps)
