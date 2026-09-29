#!/usr/bin/env python3
"""Generate predictions and export them to output/predictions.csv.

Trains the pipeline on the full training set, runs predict() on X, and
writes the resulting DataFrame to ``<PROJECT_ROOT>/output/predictions.csv``.

Usage
-----
    python scripts/predict.py [--data-dir PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.pipeline import make_pipeline
import skrub

from parkinson import PROJECT_ROOT
from parkinson.data import GROUP_COL, TARGET_COL, load_dataset, save_predictions
from sklearn.ensemble import HistGradientBoostingRegressor


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predictions to CSV.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing X_train.csv and y_train.csv (default: data/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "predictions.csv",
        help="Destination CSV file (default: output/predictions.csv).",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} …")
    X, y = load_dataset(args.data_dir)

    # Drop the grouping column before fitting — it is not a feature.
    X_fit = X.drop(columns=[GROUP_COL], errors="ignore")

    print("Fitting pipeline …")
    model = make_pipeline(
        skrub.TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )
    model.fit(X_fit, y)

    print("Generating predictions …")
    y_pred = pd.Series(model.predict(X_fit), index=X.index, name="prediction")

    # Build a tidy output DataFrame: index + patient_id + prediction + true target.
    result = pd.DataFrame(
        {
            GROUP_COL: X[GROUP_COL],
            "prediction": y_pred,
            TARGET_COL: y,
        }
    )

    out_path = save_predictions(result, args.output)
    print(f"Predictions saved to {out_path}  ({len(result)} rows)")


if __name__ == "__main__":
    main()
