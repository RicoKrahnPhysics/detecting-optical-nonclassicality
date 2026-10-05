"""Build an early/late extrapolation train/test pair from the raw TXT files."""

import argparse
from pathlib import Path

from raw_dataset import build


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw TXT -> extrapolation dataset')
    parser.add_argument('--raw-dir', type=Path, required=True,
                        help='Directory containing weak_pnr_100000_*.txt for one run')
    parser.add_argument('--run', type=int, required=True, help='Run number (1-8)')
    parser.add_argument('--train-percent', type=int, help='One early train fraction; omit for 10, 20, ..., 90')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'datasets')
    args = parser.parse_args()
    if not 1 <= args.run <= 8 or (args.train_percent is not None
                                  and args.train_percent not in range(10, 100, 10)):
        parser.error('run must be 1-8 and train-percent must be 10, 20, ..., 90')
    percents = [args.train_percent] if args.train_percent is not None else range(10, 100, 10)
    build(args.raw_dir, args.out_dir, f'Run{args.run}', 'extrapol', percents)
