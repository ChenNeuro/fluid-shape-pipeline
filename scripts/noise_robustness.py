"""Noise robustness: probe-based ML vs wake-field DL under increasing signal noise."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract.feature_engineering import extract_features_from_df
from ml.wake_common import LabelMaps, build_label_maps, stratification_labels
from sim.config import load_config
from sim.data_schema import read_metadata
from vision.wake_dataset import load_wake_bundle, variant_tensor

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
FEATURES_CSV = ROOT / "data" / "features" / "features.csv"
OUT_DIR = ROOT / "reports" / "noise_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_ESTIMATORS = 300
REPEAT_SEEDS = [42, 43, 44]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(seed: int):
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1,
                                class_weight="balanced_subsample")
    et = ExtraTreesClassifier(n_estimators=max(450, int(1.5 * N_ESTIMATORS)), random_state=seed,
                              n_jobs=-1, class_weight="balanced_subsample")
    svc = Pipeline([("scaler", StandardScaler()),
                    ("svc", SVC(C=10.0, kernel="rbf", gamma="scale"))])
    return VotingClassifier(estimators=[("rf", rf), ("et", et), ("svc", svc)], voting="hard")


def _noisy_probe_features(case_dir: Path, noise_std_ratio: float, n_bands: int = 6,
                           pod_modes: int = 5) -> dict[str, float] | None:
    """Load raw probes, add noise, re-extract features."""
    probes_path = case_dir / "probes.csv"
    if not probes_path.exists():
        return None
    df = pd.read_csv(probes_path)
    probe_cols = [c for c in df.columns if c.startswith("u_")]
    signal_matrix = df[probe_cols].to_numpy(dtype=float)

    # Add Gaussian noise scaled by per-probe std
    per_probe_std = signal_matrix.std(axis=0)
    noise_std = float(noise_std_ratio) * per_probe_std
    noise = np.random.default_rng(abs(hash(str(case_dir))) % 2**31).normal(
        0, noise_std[None, :], signal_matrix.shape
    )
    df_noisy = df.copy()
    df_noisy[probe_cols] = signal_matrix + noise

    try:
        return extract_features_from_df(df_noisy, n_bands=n_bands, pod_modes=pod_modes, add_pod=True)
    except ValueError:
        return None


def _load_wake_model(run_name: str):
    model_path = ROOT / "models" / run_name / "wake_field_main.pt"
    pkg = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if pkg["model_type"] == "jepa":
        from vision.wake_model import MultiScaleJEPAModel
        model = MultiScaleJEPAModel(**pkg["model_kwargs"]).to(DEVICE)
    else:
        from vision.wake_model import MultiScaleWakeNet
        model = MultiScaleWakeNet(**pkg["model_kwargs"]).to(DEVICE)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    return model, pkg


def _evaluate_wake_model(model, x: np.ndarray, y_shape: np.ndarray, noise_std: float,
                         batch_size: int = 32, label_map: dict | None = None) -> float:
    """Add noise to normalized wake fields, evaluate F1."""
    x_tensor = torch.from_numpy(x).float().to(DEVICE)
    if noise_std > 0:
        noise = torch.randn_like(x_tensor) * noise_std
        x_tensor = x_tensor + noise

    n = x_tensor.shape[0]
    all_preds = []
    for i in range(0, n, batch_size):
        batch = x_tensor[i:i + batch_size]
        with torch.no_grad():
            out = model(batch)
        preds = out["shape_logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
    y_pred_idx = np.concatenate(all_preds)
    if label_map is not None:
        y_pred_str = np.array([label_map[int(i)] for i in y_pred_idx])
        y_true_str = np.array(y_shape, dtype=str)
        return float(f1_score(y_true_str, y_pred_str, average="macro"))
    return float(f1_score(y_shape, y_pred_idx, average="macro"))


def main():
    cfg = load_config("configs/wake_field_1500.yaml")
    print("Loading data...")
    bundle = load_wake_bundle()
    label_maps = build_label_maps(bundle)
    strata = stratification_labels(bundle)

    # --- Probe pipeline ---
    print("=== Probe pipeline noise sweep ===")
    index_df = pd.read_csv(FEATURES_CSV)
    case_ids = index_df["case_id"].tolist()
    meta_cols = ["case_id", "shape", "Re", "dy", "eps", "seed"]
    y = index_df["shape"].to_numpy()

    probe_results: dict[float, list[float]] = {}
    for nl in NOISE_LEVELS:
        print(f"  noise_level={nl:.2f}")
        all_rows = []
        for case_id in case_ids:
            case_dir = DATA_DIR / case_id
            feats = _noisy_probe_features(case_dir, nl, n_bands=6, pod_modes=5)
            if feats is None:
                continue
            meta = read_metadata(case_dir)
            row = {"case_id": case_id, "shape": meta["shape"], "Re": int(meta["Re"]),
                   "dy": float(meta["dy"]), "eps": float(meta["eps"]), "seed": int(meta["seed"])}
            row.update(feats)
            all_rows.append(row)

        if not all_rows:
            continue
        df_feat = pd.DataFrame(all_rows)
        feat_cols = [c for c in df_feat.columns if c not in meta_cols]
        X = df_feat[feat_cols].to_numpy(dtype=float)
        y_feat = df_feat["shape"].to_numpy()
        strata_feat = stratification_labels_for_df(df_feat)

        f1s = []
        for seed in REPEAT_SEEDS:
            idx_train, idx_test = train_test_split(
                np.arange(X.shape[0]), test_size=0.2, random_state=seed, stratify=strata_feat)
            model = _build_model(seed)
            model.fit(X[idx_train], y_feat[idx_train])
            y_pred = model.predict(X[idx_test])
            f1s.append(f1_score(y_feat[idx_test], y_pred, average="macro"))
        probe_results[nl] = f1s
        print(f"    F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # --- Wake-field DL pipeline ---
    spec = {"scales": ["dist0.5_full", "dist1.0_full", "dist2.0_full"],
            "channels": ["ux", "uy", "speed", "vorticity"]}
    x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])

    # Reuse the same test idx (export_seed=42) for fair cross-method comparison
    idx_keep = np.arange(bundle.case_ids.shape[0])
    strata = stratification_labels(bundle)
    idx_train, idx_test = train_test_split(
        idx_keep, test_size=0.2, random_state=42, stratify=strata)
    print(f"Test set: {idx_test.shape[0]} cases")

    for run_name, label_str in [("jepa", "JEPA"), ("resnet18", "ResNet18")]:
        model_path = ROOT / "models" / run_name / "wake_field_main.pt"
        if not model_path.exists():
            print(f"  {run_name} not found, skip")
            continue
        print(f"\n=== {label_str} noise sweep ===")
        model, pkg = _load_wake_model(run_name)
        shape_labels = pkg["shape_labels"]

        x_test = x_all[idx_test]
        y_test = bundle.shapes[idx_test]

        for nl in NOISE_LEVELS:
            f1s = [_evaluate_wake_model(model, x_test, y_test, nl, label_map=label_maps.idx_to_shape)
                   for _ in range(3)]
            print(f"  noise={nl:.2f}  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # --- Save ---
    rows = []
    for nl, f1s in probe_results.items():
        rows.append({"method": "Probe+Ensemble", "noise_level": nl,
                     "f1_mean": np.mean(f1s), "f1_std": np.std(f1s)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "noise_robustness.csv", index=False)
    print(f"\nResults saved to {OUT_DIR / 'noise_robustness.csv'}")


def stratification_labels_for_df(df: pd.DataFrame) -> np.ndarray:
    return (df["shape"].astype(str) + "_Re" + df["Re"].astype(str)).to_numpy()


if __name__ == "__main__":
    main()
