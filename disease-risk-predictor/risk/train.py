"""Train and compare several models, keep the best one.

Usage (from the project root):
    python -m risk.train                 # trains, saves model + reports (+ plots if matplotlib is installed)
    python -m risk.train --no-plots
    python -m risk.train --data path/to/diabetes.csv

The best model is chosen on cross-validated ROC AUC computed on the TRAINING set only.
The test set is used once at the end for the final numbers, so they are not inflated.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from risk.features import FEATURES, TARGET, zeros_to_nan

SEED = 42
ROOT = Path(__file__).resolve().parent.parent


def make_pipeline(estimator):
    return Pipeline([
        ("zeros_to_nan", FunctionTransformer(zeros_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def candidates():
    return {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "SVM (RBF)": make_pipeline(
            CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, class_weight="balanced"), ensemble=False, cv=5)),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                   random_state=SEED)),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED)),
    }


def test_metrics(pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(data_path, model_path, reports_dir, plots=True):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, pipe in candidates().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {
            "cv_roc_auc_mean": round(float(scores.mean()), 4),
            "cv_roc_auc_std": round(float(scores.std()), 4),
            "test": test_metrics(pipe, X_test, y_test),
        }
        print(f"{name:20s} CV AUC {scores.mean():.3f} (+/- {scores.std():.3f})  "
              f"test AUC {results[name]['test']['roc_auc']:.3f}  "
              f"acc {results[name]['test']['accuracy']:.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    best = fitted[best_name]
    print(f"\nBest model (by cross-validation): {best_name}")

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=20, random_state=SEED)
    importance = sorted(
        ({"feature": f, "importance": round(float(m), 4)} for f, m in zip(FEATURES, imp.importances_mean)),
        key=lambda d: d["importance"], reverse=True)

    info = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {"rows": int(len(df)), "positive_rate": round(float(y.mean()), 4),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "test_metrics": results[best_name]["test"],
        "comparison": results,
        "feature_importance": importance,
    }

    model_path, reports_dir = Path(model_path), Path(reports_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "features": FEATURES, "info": info}, model_path)
    (reports_dir / "metrics.json").write_text(json.dumps(info, indent=2))
    print(f"Saved model to {model_path} and metrics to {reports_dir / 'metrics.json'}")

    if plots:
        try:
            make_plots(fitted, results, best_name, X_test, y_test, importance, reports_dir)
        except ImportError:
            print("matplotlib not installed, skipping plots (pip install -r requirements-dev.txt)")
    return info


def make_plots(fitted, results, best_name, X_test, y_test, importance, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, [results[n]["cv_roc_auc_mean"] for n in names], 0.4, label="Cross-validation (train)")
    ax.bar(x + 0.2, [results[n]["test"]["roc_auc"] for n in names], 0.4, label="Held-out test")
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=n,
                                        curve_kwargs={"lw": 2.5 if n == best_name else 1})
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curves (test set)")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=130)
    plt.close(fig)

    cm = results[best_name]["test"]["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    top = importance[::-1]
    ax.barh([d["feature"] for d in top], [d["importance"] for d in top], color="#2f7ff0")
    ax.set_xlabel("Drop in ROC AUC when the feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=130)
    plt.close(fig)
    print("Saved plots to", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "diabetes.csv"))
    p.add_argument("--model", default=str(ROOT / "model" / "model.joblib"))
    p.add_argument("--reports", default=str(ROOT / "reports"))
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    train(a.data, a.model, a.reports, plots=not a.no_plots)


if __name__ == "__main__":
    main()
