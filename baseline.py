#!/usr/bin/env python3
"""Reproducible SMILES-only baseline for SI VACV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

SEED = 42
FINGERPRINT_BITS = 2048
RADIUS = 2


def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert a SMILES string into a Morgan bit fingerprint."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=RADIUS, nBits=FINGERPRINT_BITS
    )
    vector = np.zeros((FINGERPRINT_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, vector)
    return vector


def make_features(smiles: pd.Series) -> np.ndarray:
    """Build model inputs using SMILES only; assay columns are never read here."""
    return np.vstack([smiles_to_fingerprint(value) for value in smiles])


def validate_submission(
    submission: pd.DataFrame, sample: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Raise ValueError unless submission conforms exactly to the provided template."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns/order do not match sample_submit.csv")
    if len(submission) != len(test) or len(submission) != len(sample):
        raise ValueError("Submission row count does not match test/template")
    if not submission["index"].equals(sample["index"]):
        raise ValueError("Submission index/order does not match sample_submit.csv")
    if not submission["index"].equals(test["index"]):
        raise ValueError("Submission index/order does not match test.csv")
    values = pd.to_numeric(submission["SI VACV"], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Submission predictions must all be finite numbers")


def run(data_dir: Path, output_path: Path) -> float:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submit.csv")

    required_train = {"SMILES", "SI VACV"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train.csv must contain {sorted(required_train)}")
    if "SMILES" not in test or "index" not in test:
        raise ValueError("test.csv must contain SMILES and index")
    if list(sample.columns) != ["index", "SI VACV"]:
        raise ValueError("sample_submit.csv columns must be ['index', 'SI VACV']")
    if train["SMILES"].isna().any() or test["SMILES"].isna().any():
        raise ValueError("Missing SMILES values are not supported")

    # Select only SMILES for features and SI VACV for labels. In particular,
    # CC50_uM and IC50_uM_VACV are never passed to feature construction/model.
    X = make_features(train["SMILES"])
    y = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Training target contains non-finite values")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    validation_mse = mean_squared_error(y_valid, model.predict(X_valid))
    print(f"Validation MSE (20% random split, seed={SEED}): {validation_mse:.6f}")
    print("Note: with only 320 training rows, a single split is a noisy estimate.")

    # Refit the same fixed model on all labeled training rows before test prediction.
    model.fit(X, y)
    predictions = model.predict(make_features(test["SMILES"]))
    submission = pd.DataFrame({"index": test["index"], "SI VACV": predictions})
    validate_submission(submission, sample, test)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} validated predictions to {output_path}")
    return validation_mse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    run(args.data_dir, args.output)


if __name__ == "__main__":
    main()
