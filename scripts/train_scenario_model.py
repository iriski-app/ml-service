"""Export the existing notebook's scenario CatBoost with an independent test.

The features, target and CatBoost parameters follow
notebooks/rate_sensitivity/02_linear_regression_catboost.ipynb. The test is no
longer used for early stopping: all scenarios of each bond stay in one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

FEATURES = ["shock_abs", "DurationMod", "YIELDCLOSE", "FACEVALUE"]
SHIFTS_BP = [-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]


def prepare_data(path: Path) -> pd.DataFrame:
    columns = ["SECID", "TRADEDATE", "MATDATE", "CLOSE", "DURATION", "YIELDCLOSE", "FACEVALUE"]
    data = pd.read_csv(path, usecols=columns)
    for name in ("TRADEDATE", "MATDATE"):
        data[name] = pd.to_datetime(data[name], errors="coerce")
    for name in ("CLOSE", "DURATION", "YIELDCLOSE", "FACEVALUE"):
        data[name] = pd.to_numeric(data[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=columns).copy()
    if data.duplicated(["SECID", "TRADEDATE"]).any():
        raise ValueError("duplicate SECID and trade date")
    data = data[data["YIELDCLOSE"] > -100].copy()
    data["T"] = (data["MATDATE"] - data["TRADEDATE"]).dt.days / 365.25
    data["DurationMod"] = data["DURATION"] / 365 / (1 + data["YIELDCLOSE"] / 100)
    data["Convexity"] = .75 * data["T"] * (data["T"] + 1) / (1 + data["YIELDCLOSE"] / 100) ** 2
    return data[data["T"].gt(.05) & data["DurationMod"].between(0, 20, inclusive="right")
                & data["FACEVALUE"].between(0, 1_000_000, inclusive="right")].sort_values(["SECID", "TRADEDATE"])


def split_bonds(data: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    ids = np.asarray(sorted(data["SECID"].unique()))
    if len(ids) < 10:
        raise ValueError("at least ten different bonds are required")
    np.random.default_rng(seed).shuffle(ids)
    a, b = int(.6 * len(ids)), int(.8 * len(ids))
    return {name: data[data["SECID"].isin(part)].copy()
            for name, part in zip(("train", "validation", "test"), (ids[:a], ids[a:b], ids[b:]), strict=True)}


def scenarios(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([data.assign(shift_bp=bp, shock_abs=bp / 10000) for bp in SHIFTS_BP], ignore_index=True)
    dy = result["shock_abs"]
    result["target"] = (-result["DurationMod"] * dy + .5 * result["Convexity"] * dy ** 2) * result["FACEVALUE"] * 10
    return result


def train(dataset: Path, output: Path, *, iterations: int = 500, seed: int = 42) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    frames = split_bonds(prepare_data(dataset), seed)
    expanded = {name: scenarios(frame) for name, frame in frames.items()}
    training, validation, test = (expanded[name] for name in ("train", "validation", "test"))
    model = CatBoostRegressor(iterations=iterations, learning_rate=.05, depth=6,
                              loss_function="MAE", random_seed=seed, thread_count=4,
                              allow_writing_files=False, verbose=False)
    model.fit(training[FEATURES], training["target"],
              eval_set=(validation[FEATURES], validation["target"]), early_stopping_rounds=50)
    predicted = np.asarray(model.predict(test[FEATURES]))
    zero = test["shift_bp"].eq(0)
    raw_zero_mae = float(np.abs(predicted[zero]).mean())
    predicted[zero] = 0  # Same exact-zero policy as the serving code.
    errors = predicted - test["target"].to_numpy()
    if not np.isfinite(errors).all():
        raise ValueError("non-finite held-out prediction")
    report = {
        "seed": seed, "trees": model.tree_count_, "features": FEATURES,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "target": "Synthetic duration-convexity formula with approximate convexity, ten face values.",
        "split_scope": "Held-out bonds at overlapping dates; not a future-period market forecast.",
        "zero_shift_policy": "exact_zero_by_formula", "raw_model_zero_shift_mae": raw_zero_mae,
        "tested_shifts_bp": SHIFTS_BP,
        "input_ranges": {field: [float(frames["train"][column].min()), float(frames["train"][column].max())]
                         for field, column in [("duration_mod", "DurationMod"), ("ytm", "YIELDCLOSE"), ("face_value", "FACEVALUE")]},
        "splits": {name: {"bonds": sorted(frame["SECID"].unique().tolist()), "rows": len(frame),
                          "date_min": str(frame["TRADEDATE"].min().date()), "date_max": str(frame["TRADEDATE"].max().date())}
                   for name, frame in frames.items()},
        "test_mae_rub_for_10_face_values": float(np.abs(errors).mean()),
        "test_rmse_rub_for_10_face_values": float(np.sqrt(np.mean(errors ** 2))),
        "by_shift": [{"shift_bp": bp, "mae": float(np.abs(errors[test["shift_bp"].eq(bp)]).mean())}
                     for bp in SHIFTS_BP],
    }
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output / "scenario.cbm"))
    report["model_sha256"] = hashlib.sha256((output / "scenario.cbm").read_bytes()).hexdigest()
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/scenario"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(args.dataset, args.output_dir, iterations=args.iterations, seed=args.seed)
    print(json.dumps({key: report[key] for key in ("trees", "test_mae_rub_for_10_face_values", "test_rmse_rub_for_10_face_values")}, indent=2))
