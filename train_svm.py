#!/usr/bin/env python3
"""只使用训练集训练 Linear SVM，并保存模型及 STM32 参数。

运行示例：
    python train_svm.py --input train_features.csv
    python train_svm.py --input train_features.csv \
        --features mav wl zc ssc
"""

from __future__ import annotations

# argparse：读取终端参数；json：导出 STM32 参数；Path：处理路径。
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


KNOWN_FEATURES = ("mav", "rms", "wl", "var", "zc", "ssc", "wamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a relax/bend Linear SVM on one training dataset."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Training feature CSV."
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("linear_svm.joblib")
    )
    parser.add_argument(
        "--params-output", type=Path, default=Path("linear_svm_params.json")
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=KNOWN_FEATURES,
        help="Feature columns to use. Default: all known columns found.",
    )
    # C 是正则化参数：越大越重视训练错误，过大时可能过拟合。
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def make_model(c_value: float) -> Pipeline:
    """建立 StandardScaler -> LinearSVC 训练流水线。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual=False,
            max_iter=20_000,
        )),
    ])


def validate_training_data(
    frame: pd.DataFrame,
    requested_features: list[str] | None,
) -> list[str]:
    """检查训练表并返回实际使用的特征顺序。"""
    required = {"trial_id", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    features = requested_features or [
        name for name in KNOWN_FEATURES if name in frame.columns
    ]
    if not features:
        raise ValueError("No recognized feature columns were found.")
    missing_features = set(features) - set(frame.columns)
    if missing_features:
        raise ValueError(
            f"Requested features not present: {sorted(missing_features)}"
        )
    if frame[features].isna().any().any():
        raise ValueError("Training features contain NaN values.")
    if set(frame["label_id"].unique()) != {0, 1}:
        raise ValueError(
            "Training label_id must contain both classes 0 and 1."
        )
    return features


def export_parameters(
    model: Pipeline,
    features: list[str],
    path: Path,
    c_value: float,
) -> None:
    """导出 STM32 标准化和 Linear SVM 推理所需参数。"""
    scaler: StandardScaler = model.named_steps["scaler"]
    svm: LinearSVC = model.named_steps["svm"]
    parameters = {
        "feature_order": features,
        "labels": {"0": "relax", "1": "bend"},
        "c": c_value,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "svm_weight": svm.coef_[0].tolist(),
        "svm_bias": float(svm.intercept_[0]),
        "decision_rule": (
            "z[i]=(x[i]-scaler_mean[i])/scaler_scale[i]; "
            "score=sum(svm_weight[i]*z[i])+svm_bias; "
            "score>=0 => bend, else relax"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    features = validate_training_data(frame, args.features)

    # 使用 DataFrame 保留列名；模型和验证脚本必须采用相同特征顺序。
    x = frame[features]
    y = frame["label_id"].to_numpy(dtype=np.int64)

    # 只有本文件调用 fit()；验证数据不会进入训练过程。
    model = make_model(args.c)
    model.fit(x, y)

    train_trials = int(frame["trial_id"].astype(str).nunique())
    artifact = {
        "model": model,
        "features": features,
        "c": args.c,
        "training_input": str(args.input),
        "train_windows": len(frame),
        "train_trials": train_trials,
    }

    # joblib 中同时保存模型和特征顺序，验证时无需手工指定列顺序。
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output)
    export_parameters(model, features, args.params_output, args.c)

    class_counts = frame.groupby("label_id").size().to_dict()
    print(f"Training input: {args.input}")
    print(f"Features ({len(features)}): {features}")
    print(f"Training windows: {len(frame)}, trials: {train_trials}")
    print(f"Class counts: {class_counts}")
    print(f"Saved sklearn model: {args.model_output.resolve()}")
    print(f"Saved STM32 parameters: {args.params_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
