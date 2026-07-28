#!/usr/bin/env python3
"""把队员的 ``raw.bin + bin标签CSV`` 转成当前项目需要的连续Session CSV。

原始TYQ格式：

* ``session_XXX_raw.bin``：单通道 int16 little-endian ADC samples；
* ``session_XXX_labels.csv``：每64个sample一个标签，0=Fist，1=Relax。

当前项目格式：

* ``Channel1_raw``：逐sample原始ADC值；
* ``button_label``：逐sample标签，0=Relax，1=Clench。

原始文件不会被修改或删除。没有完整bin标签的文件尾部sample会被明确丢弃，
转换统计写入 ``conversion_manifest.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TYQ raw.bin/bin-label data to continuous CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_recorded_tyq/data_recorded"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_recorded_tyq/converted"),
    )
    parser.add_argument(
        "--train-sessions",
        nargs="+",
        default=["006"],
        metavar="ID",
    )
    parser.add_argument(
        "--validation-sessions",
        nargs="+",
        default=["007"],
        metavar="ID",
    )
    parser.add_argument("--sampling-rate", type=float, default=500.0)
    parser.add_argument("--bin-samples", type=int, default=64)
    args = parser.parse_args()

    if args.sampling_rate <= 0.0:
        parser.error("--sampling-rate must be positive.")
    if args.bin_samples < 1:
        parser.error("--bin-samples must be positive.")
    overlap = set(args.train_sessions) & set(args.validation_sessions)
    if overlap:
        parser.error(
            "A session cannot be both training and validation data: "
            f"{sorted(overlap)}"
        )
    return args


def read_and_validate_labels(
    path: Path,
    raw_sample_count: int,
    bin_samples: int,
) -> pd.DataFrame:
    """读取bin标签，并严格检查索引连续性、长度和标签编码。"""
    labels = pd.read_csv(path)
    required = ["bin_index", "start_row", "end_row", "label"]
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    if labels.empty:
        raise ValueError(f"{path}: label table is empty.")

    for column in required:
        labels[column] = pd.to_numeric(labels[column], errors="raise")
        values = labels[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {column} contains NaN/Inf.")
        if not np.all(values == np.rint(values)):
            raise ValueError(f"{path}: {column} must contain integers.")
        labels[column] = values.astype(np.int64)

    expected_bin_index = np.arange(len(labels), dtype=np.int64)
    if not np.array_equal(
        labels["bin_index"].to_numpy(), expected_bin_index
    ):
        raise ValueError(f"{path}: bin_index must start at 0 and be contiguous.")

    starts = labels["start_row"].to_numpy(dtype=np.int64)
    ends = labels["end_row"].to_numpy(dtype=np.int64)
    if starts[0] != 0:
        raise ValueError(f"{path}: the first labelled sample must be row 0.")
    if np.any(starts < 0) or np.any(ends < starts):
        raise ValueError(f"{path}: invalid start_row/end_row.")
    if not np.all(ends - starts + 1 == bin_samples):
        raise ValueError(
            f"{path}: every label bin must contain {bin_samples} samples."
        )
    if len(labels) > 1 and not np.array_equal(starts[1:], ends[:-1] + 1):
        raise ValueError(f"{path}: label bins contain a gap or overlap.")
    if int(ends[-1]) >= raw_sample_count:
        raise ValueError(
            f"{path}: labels require sample {int(ends[-1])}, but raw file "
            f"contains only {raw_sample_count} samples."
        )

    unique_labels = set(labels["label"].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{path}: source labels must be 0=Fist or 1=Relax; "
            f"found {sorted(unique_labels)}."
        )
    if unique_labels != {0, 1}:
        raise ValueError(f"{path}: both Fist and Relax labels are required.")
    return labels


def convert_session(
    session_id: str,
    split: str,
    input_dir: Path,
    output_root: Path,
    sampling_rate: float,
    bin_samples: int,
) -> dict[str, Any]:
    """转换一个session，并返回可写入manifest的统计信息。"""
    raw_path = input_dir / f"session_{session_id}_raw.bin"
    label_path = input_dir / f"session_{session_id}_labels.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if raw_path.stat().st_size % np.dtype("<i2").itemsize:
        raise ValueError(f"{raw_path}: byte count is not divisible by int16.")

    raw = np.fromfile(raw_path, dtype="<i2")
    labels = read_and_validate_labels(
        label_path,
        raw_sample_count=raw.size,
        bin_samples=bin_samples,
    )
    labelled_sample_count = int(labels["end_row"].iloc[-1]) + 1

    source_label = np.empty(labelled_sample_count, dtype=np.uint8)
    for row in labels.itertuples(index=False):
        start = int(row.start_row)
        stop = int(row.end_row) + 1  # end_row在原文件中是inclusive。
        source_label[start:stop] = int(row.label)

    # 队员：0=Fist、1=Relax；本项目：0=Relax、1=Clench。
    button_label = (1 - source_label).astype(np.uint8)
    sample_index = np.arange(labelled_sample_count, dtype=np.int64)
    converted = pd.DataFrame(
        {
            "frame_index": sample_index,
            "recording_sample_index": sample_index,
            "recording_elapsed_s": sample_index / sampling_rate,
            "button_label": button_label,
            "Channel1_raw": raw[:labelled_sample_count],
        }
    )

    destination_dir = output_root / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"session_{session_id}.csv"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite converted CSV: {destination}"
        )
    temporary = destination.with_suffix(".csv.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary conversion file already exists: {temporary}"
        )
    converted.to_csv(temporary, index=False, float_format="%.6f")
    temporary.rename(destination)

    current_counts = converted["button_label"].value_counts().sort_index()
    return {
        "session_id": session_id,
        "split": split,
        "source_raw": str(raw_path.resolve()),
        "source_labels": str(label_path.resolve()),
        "converted_csv": str(destination.resolve()),
        "raw_samples": int(raw.size),
        "labelled_samples": labelled_sample_count,
        "discarded_unlabelled_tail_samples": int(
            raw.size - labelled_sample_count
        ),
        "duration_s": labelled_sample_count / sampling_rate,
        "source_bins": int(len(labels)),
        "current_label_sample_counts": {
            "0_relax": int(current_counts.get(0, 0)),
            "1_clench": int(current_counts.get(1, 0)),
        },
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sessions: list[dict[str, Any]] = []
    for split, session_ids in (
        ("train", args.train_sessions),
        ("validation", args.validation_sessions),
    ):
        for session_id in session_ids:
            result = convert_session(
                session_id=session_id,
                split=split,
                input_dir=input_dir,
                output_root=output_root,
                sampling_rate=args.sampling_rate,
                bin_samples=args.bin_samples,
            )
            sessions.append(result)
            print(
                f"Converted session {session_id} -> {split}: "
                f"{result['labelled_samples']} samples, "
                f"dropped tail={result['discarded_unlabelled_tail_samples']}"
            )

    manifest = {
        "format": "tyq_bin_labels_to_continuous_session_v1",
        "source_sampling_rate_hz": args.sampling_rate,
        "source_bin_samples": args.bin_samples,
        "source_bin_duration_ms": (
            1000.0 * args.bin_samples / args.sampling_rate
        ),
        "label_mapping": {
            "source": {"0": "fist", "1": "relax"},
            "converted": {"0": "relax", "1": "clench"},
            "formula": "converted_button_label = 1 - source_label",
        },
        "warning": (
            "Source labels were assigned once per completed bin, so an edge "
            "has up to one bin of timing uncertainty. Use a conservative "
            "transition guard during feature extraction."
        ),
        "sessions": sessions,
    }
    manifest_path = output_root / "conversion_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite conversion manifest: {manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
