"""Compact JSONL inference logs into partitioned Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compact_logs(log_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for log_file in sorted(log_dir.glob("*.jsonl")):
        df = pd.read_json(log_file, lines=True)
        if df.empty:
            continue
        day = log_file.stem
        df.to_parquet(output_dir / f"logs_{day}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact JSONL logs to Parquet")
    parser.add_argument(
        "--log-dir", default="data/logs", help="Directory containing JSONL logs"
    )
    parser.add_argument(
        "--output-dir", default="data/metrics/daily", help="Destination directory"
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if not log_dir.exists():
        raise SystemExit(f"Log directory {log_dir} does not exist")

    compact_logs(log_dir, output_dir)
    print(f"Compacted logs written to {output_dir.resolve()}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
