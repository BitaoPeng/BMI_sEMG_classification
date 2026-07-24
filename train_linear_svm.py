#!/usr/bin/env python3
"""Train and evaluate a Linear SVM from emg_feature_extractor.py output.

With ``--validation-input``, the first CSV is used only for training and the
second CSV only for independent validation. Without it, grouped
cross-validation is performed by trial_id, not by individual windows. This
prevents overlapping windows from the same recording leaking across a split.

Example:
    python train_linear_svm.py --input emg_features.csv
    python train_linear_svm.py --input train.csv --validation-input val.csv
    python train_linear_svm.py --features mav wl zc ssc
"""

from __future__ import annotations

# argparse：读取终端参数；
# json：生成 STM32 参数文件；
# Path：处理文件路径。
import argparse  
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# sklearn: 机器学习模块
# accuracy_score 等：模型评价指标
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# StratifiedGroupKFold：分组交叉验证
# StandardScaler：特征标准化
# LinearSVC：线性 SVM
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


KNOWN_FEATURES = ("mav", "rms", "wl", "var", "zc", "ssc", "wamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a trial-grouped relax/bend Linear SVM."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("emg_features.csv")
    )
    parser.add_argument(
        "--validation-input",
        type=Path,
        help=(
            "Optional independent validation feature CSV. When provided, "
            "--input is used only for training and this file only for "
            "validation."
        ),
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("linear_svm.joblib")
    )
    parser.add_argument(
        "--params-output", type=Path, default=Path("linear_svm_params.json")
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON file for cross-validation metrics.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=KNOWN_FEATURES,
        help="Feature columns to use. Default: all known columns found.",
    )
    # C 是 SVM 的正则化参数（regularization parameter）
    #   较小的 C：允许一些训练错误，分类边界更简单；
    #   较大的 C：更努力正确分类训练数据，但可能过拟合；
    #   默认 C=1.0：适合先作为基准。
    parser.add_argument("--c", type=float, default=1.0)
    # 进行5折交叉验证（5-fold cross-validation）
    #   5折交叉验证会把20个trails分成5 groups，每组4个trial。
    #   fold 1: 
    #     training: group 2, 3, 4, 5
    #     validation: group 1
    #   fold 2: 
    #     training: group 1, 3, 4, 5
    #     validation: group 2
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of trial-grouped cross-validation folds."
    )
    # 控制随机分组。
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# 建立一条机器学习流水线pipeline：
# 原始特征 -> StandardScaler -> LinearSVC -> 预测结果
def make_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),  # 特征标准化 standardization
        ("svm", LinearSVC( 
            C=c_value, # 控制分类错误和模型简单程度之间的平衡。
            class_weight="balanced", # 如果两类窗口数量不同，自动提高较少类别的重要程度。例如：relax (500 windows), bend (400 windows),模型不会因为relax更多，就总是倾向于预测relax
            dual=False,
            max_iter=20_000, # 训练算法最多迭代20000次，降低模型没有converge的概率。
        )),
    ])


def validate_data(
    frame: pd.DataFrame,
    requested_features: list[str] | None,
) -> list[str]:
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
        raise ValueError("Feature table contains NaN values.")
    if set(frame["label_id"].unique()) != {0, 1}: # 输入必须同时包含：0：relax；1：bend (如果只包含一种动作，SVM无法学习两类之间的边界)
        raise ValueError("label_id must contain both classes 0 and 1.")
    return features


def export_parameters( # 导出STM32 parameters
    model: Pipeline,
    features: list[str],
    path: Path,
) -> None:
    """Export the scaler and Linear SVM parameters needed by STM32."""
    scaler: StandardScaler = model.named_steps["scaler"]
    svm: LinearSVC = model.named_steps["svm"]
    parameters = {
        "feature_order": features,
        "labels": {"0": "relax", "1": "bend"},
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
    features = validate_data(frame, args.features)

    x = frame[features].to_numpy(dtype=np.float64) # x: 输入特征：窗口数量 × 特征数量
    y = frame["label_id"].to_numpy(dtype=np.int64) # y: 正确答案 = [0/1, 0/1, 0/1...]
    groups = frame["trial_id"].astype(str).to_numpy() # group：trail_id (它保证同一个 CSV 文件的窗口不会同时出现在训练集和测试集中)

    # 如果提供独立验证集，只使用训练集拟合 StandardScaler 和 SVM。
    # 验证集不参与 mean、scale、weight 或 bias 的学习。
    if args.validation_input is not None:
        validation_frame = pd.read_csv(args.validation_input)
        validation_features = validate_data(validation_frame, features)
        if validation_features != features:
            raise ValueError(
                "Training and validation feature order must be identical."
            )

        validation_x = validation_frame[features].to_numpy(dtype=np.float64)
        validation_y = validation_frame["label_id"].to_numpy(dtype=np.int64)
        validation_groups = (
            validation_frame["trial_id"].astype(str).to_numpy()
        )

        model = make_model(args.c)
        model.fit(x, y)
        prediction = model.predict(validation_x)

        accuracy = accuracy_score(validation_y, prediction)
        balanced = balanced_accuracy_score(validation_y, prediction)
        f1 = f1_score(validation_y, prediction)
        matrix = confusion_matrix(
            validation_y, prediction, labels=[0, 1]
        )

        print(f"Training input: {args.input}")
        print(f"Validation input: {args.validation_input}")
        print(f"Features ({len(features)}): {features}")
        print(
            f"Training windows: {len(frame)}, "
            f"trials: {len(np.unique(groups))}"
        )
        print(
            f"Validation windows: {len(validation_frame)}, "
            f"trials: {len(np.unique(validation_groups))}"
        )
        print("\nIndependent validation metrics")
        print(f"accuracy: {accuracy:.4f}")
        print(f"balanced_accuracy: {balanced:.4f}")
        print(f"F1: {f1:.4f}")
        print("\nValidation confusion matrix [relax, bend]:")
        print(matrix)
        print("\nValidation classification report:")
        print(classification_report(
            validation_y,
            prediction,
            labels=[0, 1],
            target_names=["relax", "bend"],
            digits=4,
            zero_division=0,
        ))

        if args.metrics_output is not None:
            metrics = {
                "evaluation_mode": "independent_validation",
                "training_input": str(args.input),
                "validation_input": str(args.validation_input),
                "features": features,
                "c": args.c,
                "train_windows": len(frame),
                "train_trials": len(np.unique(groups)),
                "validation_windows": len(validation_frame),
                "validation_trials": len(np.unique(validation_groups)),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced),
                "f1": float(f1),
                "confusion_matrix": matrix.tolist(),
            }
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # 独立验证完成后保存的模型仍然只在训练集上训练；
        # 不能再用验证集重新拟合，否则验证集会变成训练数据。
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output)
        export_parameters(model, features, args.params_output)
        print(f"Saved sklearn model: {args.model_output.resolve()}")
        print(f"Saved STM32 parameters: {args.params_output.resolve()}")
        if args.metrics_output is not None:
            print(f"Saved validation metrics: {args.metrics_output.resolve()}")
        return 0

    trials_per_class = (
        frame[["trial_id", "label_id"]]
        .drop_duplicates()
        .groupby("label_id")
        .size()
    )
    max_folds = int(trials_per_class.min())
    if not 2 <= args.folds <= max_folds:
        raise ValueError(
            f"--folds must be between 2 and {max_folds} for this dataset."
        )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_metrics: list[tuple[float, float, float]] = []
    all_true: list[int] = []
    all_pred: list[int] = []

    print(f"Input: {args.input}")
    print(f"Features ({len(features)}): {features}")
    print(f"Windows: {len(frame)}, trials: {len(np.unique(groups))}")

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        model = make_model(args.c)
        model.fit(x[train_index], y[train_index])
        prediction = model.predict(x[test_index])

        accuracy = accuracy_score(y[test_index], prediction)
        balanced = balanced_accuracy_score(y[test_index], prediction)
        f1 = f1_score(y[test_index], prediction)
        fold_metrics.append((accuracy, balanced, f1))
        all_true.extend(y[test_index].tolist())
        all_pred.extend(prediction.tolist())
        test_trials = len(np.unique(groups[test_index]))
        print(
            f"Fold {fold}: trials={test_trials}, accuracy={accuracy:.4f}, "
            f"balanced_accuracy={balanced:.4f}, F1={f1:.4f}"
        )

    metric_array = np.asarray(fold_metrics)
    print("\nCross-validation mean ± standard deviation")
    for column, name in enumerate(("accuracy", "balanced_accuracy", "F1")):
        print(
            f"{name}: {metric_array[:, column].mean():.4f} "
            f"± {metric_array[:, column].std(ddof=1):.4f}"
        )
    print("\nCombined out-of-fold confusion matrix [relax, bend]:")
    print(confusion_matrix(all_true, all_pred, labels=[0, 1]))
    print("\nCombined out-of-fold classification report:")
    print(classification_report(
        all_true,
        all_pred,
        labels=[0, 1],
        target_names=["relax", "bend"],
        digits=4,
        zero_division=0,
    ))

    if args.metrics_output is not None:
        metrics = {
            "evaluation_mode": "grouped_cross_validation",
            "input": str(args.input),
            "features": features,
            "c": args.c,
            "folds": args.folds,
            "seed": args.seed,
            "windows": len(frame),
            "trials": len(np.unique(groups)),
            "fold_metrics": [
                {
                    "fold": index,
                    "accuracy": values[0],
                    "balanced_accuracy": values[1],
                    "f1": values[2],
                }
                for index, values in enumerate(fold_metrics, start=1)
            ],
            "accuracy_mean": float(metric_array[:, 0].mean()),
            "accuracy_std": float(metric_array[:, 0].std(ddof=1)),
            "balanced_accuracy_mean": float(metric_array[:, 1].mean()),
            "balanced_accuracy_std": float(metric_array[:, 1].std(ddof=1)),
            "f1_mean": float(metric_array[:, 2].mean()),
            "f1_std": float(metric_array[:, 2].std(ddof=1)),
            "confusion_matrix": confusion_matrix(
                all_true, all_pred, labels=[0, 1]
            ).tolist(),
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Cross-validation above estimates generalization.  Refit on all available
    # trials only after evaluation, producing the model intended for deployment.
    final_model = make_model(args.c)
    final_model.fit(x, y)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_output)
    export_parameters(final_model, features, args.params_output)

    print(f"Saved sklearn model: {args.model_output.resolve()}")
    print(f"Saved STM32 parameters: {args.params_output.resolve()}")
    if args.metrics_output is not None:
        print(f"Saved CV metrics: {args.metrics_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
