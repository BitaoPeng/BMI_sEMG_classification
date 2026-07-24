#!/usr/bin/env python3
"""Extract deployment-friendly time-domain features from Channel1_raw.

Input files must be named ``relax_*.csv`` or ``bend_*.csv``.  Each file is
treated as one independent trial, which is important for group-wise splitting
during SVM evaluation.

Example:
    python emg_feature_extractor.py
    python emg_feature_extractor.py --target-fs 250 --window-ms 200 \
        --overlap 0.5 --features mav rms wl zc ssc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, tf2sos


LABEL_TO_ID = {"relax": 0, "bend": 1}
AVAILABLE_FEATURES = ("mav", "rms", "wl", "var", "zc", "ssc", "wamp")


def infer_label(path: Path) -> str:
    label = path.stem.split("_", 1)[0].lower()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Cannot infer relax/bend label from {path.name}")
    return label


def design_filter(
    fs: float,
    low_hz: float,
    high_hz: float,
    notch_hz: float | None,
    order: int,
) -> np.ndarray:
    """Return cascaded second-order sections for causal MCU-like filtering."""
    nyquist = fs / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Require 0 < low < high < Nyquist ({nyquist:g} Hz), got "
            f"{low_hz:g} and {high_hz:g} Hz."
        )

    sections = [butter(order, [low_hz, high_hz], btype="bandpass",
                       fs=fs, output="sos")]
    if notch_hz is not None:
        if not 0 < notch_hz < nyquist:
            raise ValueError("Notch frequency must be below Nyquist.")
        b, a = iirnotch(notch_hz, Q=30.0, fs=fs)
        sections.append(tf2sos(b, a))
    return np.vstack(sections)


def preprocess(
    signal: np.ndarray,
    source_fs: int,
    target_fs: int,
    low_hz: float,
    high_hz: float,
    notch_hz: float | None,
    filter_order: int,
) -> np.ndarray:
    """Anti-alias, integer-decimate, then band-pass/notch filter causally."""
    if source_fs % target_fs:
        raise ValueError(
            "For exact STM32 reproduction, target_fs must divide source_fs "
            f"exactly; got {source_fs}/{target_fs}."
        )
    factor = source_fs // target_fs
    x = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or infinite values.")

    # A causal low-pass before sample dropping prevents aliasing.  This simple
    # IIR + decimation scheme is straightforward to reproduce with CMSIS-DSP.
    if factor > 1:
        anti_alias_cutoff = 0.45 * target_fs
        anti_alias_sos = butter(
            filter_order, anti_alias_cutoff, btype="lowpass",
            fs=source_fs, output="sos"
        )
        x = sosfilt(anti_alias_sos, x)
        x = x[::factor]

    effective_high = min(high_hz, 0.45 * target_fs)
    if effective_high <= low_hz:
        raise ValueError(
            f"target_fs={target_fs} is too low for low_hz={low_hz:g}. "
            "Increase target_fs or reduce the band-pass lower cutoff."
        )
    sos = design_filter(
        target_fs, low_hz, effective_high, notch_hz, filter_order
    )
    return sosfilt(sos, x)


def threshold_crossings(values: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(
        ((values[:-1] >= 0) != (values[1:] >= 0))
        & (np.abs(values[1:] - values[:-1]) >= threshold)
    ))


def extract_features(
    window: np.ndarray,
    names: list[str],
    threshold: float,
) -> dict[str, float]:
    """Calculate low-cost time-domain features for one signal window."""
    x = np.asarray(window, dtype=np.float64)
    dx = np.diff(x)
    results: dict[str, float] = {}

    for name in names:
        if name == "mav":
            results[name] = float(np.mean(np.abs(x)))
        elif name == "rms":
            results[name] = float(np.sqrt(np.mean(x * x)))
        elif name == "wl":
            results[name] = float(np.sum(np.abs(dx)))
        elif name == "var":
            results[name] = float(np.var(x, ddof=1))
        elif name == "zc":
            results[name] = float(threshold_crossings(x, threshold))
        elif name == "ssc":
            slopes = dx
            results[name] = float(np.count_nonzero(
                (slopes[:-1] * slopes[1:] < 0)
                & (np.abs(slopes[:-1] - slopes[1:]) >= threshold)
            ))
        elif name == "wamp":
            results[name] = float(np.count_nonzero(np.abs(dx) >= threshold))
        else:
            raise ValueError(f"Unsupported feature: {name}")
    return results


def process_file(path: Path, args: argparse.Namespace) -> list[dict]:
    frame = pd.read_csv(path, usecols=[args.channel])
    raw = frame[args.channel].to_numpy(dtype=np.float64)
    filtered = preprocess(
        raw,
        source_fs=args.source_fs,
        target_fs=args.target_fs,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        notch_hz=args.notch_hz,
        filter_order=args.filter_order,
    )

    window_samples = round(args.window_ms * args.target_fs / 1000.0)
    step_samples = round(window_samples * (1.0 - args.overlap))
    first_sample = round(args.discard_ms * args.target_fs / 1000.0)
    if window_samples < 2 or step_samples < 1:
        raise ValueError("Window/overlap produces an invalid window step.")

    label = infer_label(path)
    rows: list[dict] = []
    for window_id, start in enumerate(
        range(first_sample, len(filtered) - window_samples + 1, step_samples)
    ):
        stop = start + window_samples
        features = extract_features(
            filtered[start:stop], args.features, args.threshold
        )
        rows.append({
            "source_file": path.name,
            "trial_id": path.stem,
            "window_id": window_id,
            "start_sample": start,
            "end_sample": stop,
            "start_s": start / args.target_fs,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            **features,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a window-level Channel1_raw feature dataset."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data_saved/data_saved")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("emg_features.csv")
    )
    parser.add_argument("--channel", default="Channel1_raw")
    parser.add_argument("--source-fs", type=int, default=500)
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument(
        "--discard-ms", type=float, default=300.0,
        help="Discard each trial's initial causal-filter transient."
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Window overlap fraction in [0, 1), e.g. 0.5."
    )
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=100.0)
    parser.add_argument(
        "--notch-hz", type=float, default=50.0,
        help="Power-line notch; use 0 to disable."
    )
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument(
        "--features", nargs="+", choices=AVAILABLE_FEATURES,
        default=["mav", "rms", "wl", "zc", "ssc"]
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="Noise threshold in filtered ADC counts for ZC/SSC/WAMP."
    )
    args = parser.parse_args()
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1).")
    if args.source_fs <= 0 or args.target_fs <= 0:
        parser.error("Sampling rates must be positive.")
    if args.notch_hz == 0:
        args.notch_hz = None
    return args


def main() -> int:
    args = parse_args()
    paths = sorted(
        p for pattern in ("relax_*.csv", "bend_*.csv")
        for p in args.input_dir.glob(pattern)
    )
    if not paths:
        raise FileNotFoundError(
            f"No relax_*.csv or bend_*.csv files in {args.input_dir}"
        )

    rows = [row for path in paths for row in process_file(path, args)]
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    counts = result.groupby("label").size().to_dict()
    print(f"Processed {len(paths)} trials -> {len(result)} windows")
    print(f"Class counts: {counts}")
    print(f"Features: {args.features}")
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
