"""Package artifacts produced by notebooks/price_imputation/03_catboost.ipynb.

Requires pandas and pyarrow. This script never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def package(models_dir: Path, dataset: Path, splits: Path, output_dir: Path, version: str) -> None:
    metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    required = {"duration_mod", "convexity"}
    if set(metadata["model_files"]) != required or set(metadata["interval_model_files"]) != required:
        raise ValueError("metadata must name both point and interval models")
    frame = pd.read_parquet(dataset, columns=["SECID", "trade_date", "years_to_yield_date", "currency_asof"])
    split_frame = pd.read_parquet(splits, columns=["SECID", "trade_date", "split"])
    if split_frame.duplicated(["SECID", "trade_date"]).any():
        raise ValueError("duplicate split keys")
    train_keys = split_frame.loc[split_frame["split"].eq("train"), ["SECID", "trade_date"]]
    train = frame.merge(train_keys, on=["SECID", "trade_date"], how="inner", validate="many_to_one")
    maturities = pd.to_numeric(train["years_to_yield_date"], errors="coerce").dropna()
    currencies = sorted(train["currency_asof"].dropna().astype(str).unique().tolist())
    if maturities.empty or not currencies or not version.strip():
        raise ValueError("training domain or version is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = ["metadata.json", *metadata["model_files"].values(), *metadata["interval_model_files"].values()]
    checksums = {}
    for name in files:
        source, target = models_dir / name, output_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        checksums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    deployment = {
        "model_version": version,
        "supported_currencies": currencies,
        "years_to_yield_date_min": float(maturities.min()),
        "years_to_yield_date_max": float(maturities.max()),
        "sha256": checksums,
        "limitations": [
            "Only fixed or zero-coupon, non-structured bonds with exact settlement calendar are supported.",
            "Estimates require review; historical interval coverage was below its nominal target.",
        ],
    }
    (output_dir / "deployment.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    package(arguments.models_dir, arguments.dataset, arguments.splits, arguments.output_dir, arguments.version)
