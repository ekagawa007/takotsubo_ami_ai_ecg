#!/usr/bin/env python3
# coding: utf-8
"""
層別ブロック無作為化の割付表（1ファイル）を生成する。

層:
  A: cardiologist, career >= threshold
  B: cardiologist, career < threshold
  C: non-cardiologist (profession 2-5), career >= threshold
  D: non-cardiologist, career < threshold

各層は block size 3 固定 (1:1:1) のブロックを並べる。
"""

import argparse
import csv
import os
import random

STRATA = ["A", "B", "C", "D"]
BLOCK_SIZE = 3
DEFAULT_OUTPUT_DIR = "/path/to/outputs/human_test/stratified_block_random"
DEFAULT_TABLE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "randomization_table.csv")
DEFAULT_STATE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "randomization_state.csv")


def generate_blocks(num_blocks, rng):
    blocks = []
    for block_id in range(1, num_blocks + 1):
        groups = [0, 1, 2]
        rng.shuffle(groups)
        for position_in_block, group in enumerate(groups, start=1):
            blocks.append(
                {
                    "block_id": block_id,
                    "position_in_block": position_in_block,
                    "group": group,
                }
            )
    return blocks


def write_randomization_table(output_path, blocks_per_stratum, seed):
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for stratum in STRATA:
        stratum_rng = random.Random(rng.randint(0, 2**31 - 1))
        block_order = list(range(1, blocks_per_stratum + 1))
        stratum_rng.shuffle(block_order)

        slot_id = 0
        for block_id in block_order:
            groups = [0, 1, 2]
            stratum_rng.shuffle(groups)
            for position_in_block, group in enumerate(groups, start=1):
                slot_id += 1
                rows.append(
                    {
                        "stratum": stratum,
                        "slot_id": slot_id,
                        "block_id": block_id,
                        "position_in_block": position_in_block,
                        "group": group,
                    }
                )

    temp_path = output_path + ".tmp"
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stratum", "slot_id", "block_id", "position_in_block", "group"],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, output_path)
    return len(rows)


def write_initial_state(state_path):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    temp_path = state_path + ".tmp"
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["total_assigned", 0])
        for stratum in STRATA:
            writer.writerow([stratum, 1])
    os.replace(temp_path, state_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified block randomization table (single CSV file)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TABLE_PATH,
        help=f"Output table CSV path (default: {DEFAULT_TABLE_PATH})",
    )
    parser.add_argument(
        "--state-output",
        type=str,
        default=DEFAULT_STATE_PATH,
        help=f"Initial state CSV path (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--blocks-per-stratum",
        type=int,
        default=16,
        help="Number of blocks per stratum (default: 16 -> 48 slots per stratum)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for table generation (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing table/state files",
    )
    args = parser.parse_args()

    if args.blocks_per_stratum < 1:
        raise ValueError("blocks-per-stratum must be >= 1")

    for path, label in [(args.output, "table"), (args.state_output, "state")]:
        if os.path.exists(path) and not args.force:
            raise FileExistsError(
                f"{label} file already exists: {path}. Use --force to overwrite."
            )

    num_rows = write_randomization_table(args.output, args.blocks_per_stratum, args.seed)
    write_initial_state(args.state_output)

    slots_per_stratum = args.blocks_per_stratum * BLOCK_SIZE
    print(f"Generated randomization table: {args.output}")
    print(f"  strata: {', '.join(STRATA)}")
    print(f"  blocks per stratum: {args.blocks_per_stratum}")
    print(f"  slots per stratum: {slots_per_stratum}")
    print(f"  total rows: {num_rows}")
    print(f"  seed: {args.seed}")
    print(f"Initial state file: {args.state_output}")


if __name__ == "__main__":
    main()
