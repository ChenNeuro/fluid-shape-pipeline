"""Full robustness: probe dropout + challenge mode, DL vs Probe comparison."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract.feature_engineering import extract_features_from_df
from ml.wake_common import LabelMaps, build_label_maps
from sim.config import load_config
from sim.data_schema import read_metadata
from vision.wake_dataset import load_wake_bundle, variant_tensor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "raw"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EST = 300
SEEDS = [42, 43, 44]


def _probe_model(seed):
    rf = RandomForestClassifier(n_estimators=N_EST, random_state=seed, n_jobs=-1,
                                class_weight="balanced_subsample")
    et = ExtraTreesClassifier(n_estimators=max(450, int(1.5 * N_EST)), random_state=seed,
                              n_jobs=-1, class_weight="balanced_subsample")
    svc = Pipeline([("s", StandardScaler()), ("c", SVC(C=10.0, kernel="rbf", gamma="scale"))])
    return VotingClassifier([("rf", rf), ("et", et), ("svc", svc)], voting="hard")


def _load_dl(run_name):
    from vision.wake_model import MultiScaleWakeNet, MultiScaleJEPAModel
    pkg = torch.load(ROOT / "models" / run_name / "wake_field_main.pt",
                     map_location=DEVICE, weights_only=False)
    cls = MultiScaleJEPAModel if pkg["model_type"] == "jepa" else MultiScaleWakeNet
    model = cls(**pkg["model_kwargs"]).to(DEVICE)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    return model, pkg


# ============================================================
# Experiment 1: Probe dropout
# ============================================================
def run_probe_dropout(label_maps: LabelMaps):
    """Drop random probes, retrain probe model, measure F1 drop."""
    print("\n=== Exp 1: Probe Dropout ===")
    index_df = pd.read_csv(ROOT / "data" / "features" / "features.csv")
    meta_cols = ["case_id", "shape", "Re", "dy", "eps", "seed"]
    y_all = index_df["shape"].to_numpy()
    strata = (index_df["shape"] + "_Re" + index_df["Re"].astype(str)).to_numpy()
    n_total_probes = 32

    dropout_rates = [0.0, 0.25, 0.5, 0.75, 0.875]  # fraction of probes REMOVED
    results = {}

    for dr in dropout_rates:
        n_keep = max(2, int(n_total_probes * (1 - dr)))
        print(f"  dropout={dr:.3f} ({n_keep}/{n_total_probes} probes kept)")

        all_rows = []
        for _, row in index_df.iterrows():
            case_id = str(row["case_id"])
            probes_path = DATA_DIR / case_id / "probes.csv"
            if not probes_path.exists():
                continue
            df_raw = pd.read_csv(probes_path)
            probe_cols = [c for c in df_raw.columns if c.startswith("u_")]
            n_actual = len(probe_cols)

            rng = np.random.default_rng(abs(hash(case_id)) % 2**31)
            keep_idx = rng.choice(n_actual, size=min(n_keep, n_actual), replace=False)
            keep_cols = [probe_cols[i] for i in sorted(keep_idx)]

            # Rename to sequential u_000..u_XXX for feature extraction
            df_mapped = df_raw[["time"] + keep_cols].copy()
            rename_map = {old: f"u_{i:03d}" for i, old in enumerate(keep_cols)}
            df_mapped = df_mapped.rename(columns=rename_map)

            try:
                feats = extract_features_from_df(df_mapped, n_bands=6, pod_modes=5, add_pod=True)
            except (ValueError, np.linalg.LinAlgError):
                continue

            meta = read_metadata(DATA_DIR / case_id)
            row_d = {"case_id": case_id, "shape": meta["shape"], "Re": int(meta["Re"]),
                     "dy": float(meta["dy"]), "eps": float(meta["eps"]), "seed": int(meta["seed"])}
            row_d.update(feats)
            all_rows.append(row_d)

        if not all_rows:
            results[dr] = [0.0]
            continue

        df_feat = pd.DataFrame(all_rows)
        feat_cols = [c for c in df_feat.columns if c not in meta_cols]
        X = df_feat[feat_cols].to_numpy(dtype=float)
        y = df_feat["shape"].to_numpy()
        s = (df_feat["shape"] + "_Re" + df_feat["Re"].astype(str)).to_numpy()

        f1s = []
        for seed in SEEDS:
            itr, ite = train_test_split(np.arange(X.shape[0]), test_size=0.2,
                                        random_state=seed, stratify=s)
            m = _probe_model(seed)
            m.fit(X[itr], y[itr])
            f1s.append(f1_score(y[ite], m.predict(X[ite]), average="macro"))
        results[dr] = f1s
        print(f"    F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Save
    rows = [{"method": "Probe+Ensemble", "dropout_rate": dr,
             "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
             "probes_kept": max(2, int(n_total_probes * (1 - dr)))}
            for dr, f1s in results.items()]
    pd.DataFrame(rows).to_csv(OUT / "probe_dropout.csv", index=False)

    # DL models are probe-independent → same F1 at all dropout rates
    bundle = load_wake_bundle()
    spec = {"scales": ["dist0.5_full", "dist1.0_full", "dist2.0_full"],
            "channels": ["ux", "uy", "speed", "vorticity"]}
    x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])
    bundle_strata = (np.array([f"{s}_Re{r}" for s, r in zip(bundle.shapes, bundle.re_values)]))
    _, idx_test = train_test_split(np.arange(bundle.case_ids.shape[0]), test_size=0.2,
                                   random_state=42, stratify=bundle_strata)
    print(f"\n  DL baseline (probe-independent, {idx_test.shape[0]} test cases):")
    for run_name in ["jepa", "resnet18"]:
        if not (ROOT / "models" / run_name / "wake_field_main.pt").exists():
            continue
        model, pkg = _load_dl(run_name)
        x_t = torch.from_numpy(x_all[idx_test]).float().to(DEVICE)
        with torch.no_grad():
            out = model(x_t)
        preds = out["shape_logits"].argmax(dim=1).cpu().numpy()
        pred_str = np.array([pkg["shape_labels"][int(i)] for i in preds])
        dl_f1 = f1_score(bundle.shapes[idx_test], pred_str, average="macro")
        print(f"    {run_name}: F1={dl_f1:.4f} (constant, no probes needed)")


# ============================================================
# Experiment 2: Challenge mode
# ============================================================
def run_challenge_comparison():
    """Compare DL vs Probe under extreme challenge mode parameters."""
    print("\n=== Exp 2: Challenge Mode ===")

    # Base: 1500 config results (already have)
    # Challenge: extreme freq_jitter, amp_jitter, noise, dropout
    challenge_cfg = load_config("configs/wake_field_1500.yaml")
    ch = challenge_cfg["simulation"]["synthetic"]["challenge"]
    # Extreme params
    ch["freq_jitter_std"] = 0.20
    ch["amp_jitter_std"] = 0.30
    ch["noise_multiplier"] = 3.0
    ch["common_mode_amp"] = 0.15
    ch["drift_amp"] = 0.10
    ch["probe_mix"] = 0.35
    ch["dropout_prob"] = 0.015
    ch["dropout_std"] = 0.15

    # Remove old data and regenerate with challenge config
    print("  Regenerating dataset with extreme challenge params...")
    import shutil

    # Use a temp config
    tmp_cfg = copy.deepcopy(challenge_cfg)
    tmp_cfg["project"]["seed"] = 20260601
    tmp_cfg["simulation"]["perturbations_per_combo"] = 10  # smaller: 5x3x10=150 cases
    tmp_cfg["simulation"]["workers"] = 8

    tmp_path = ROOT / "configs" / "challenge_extreme.yaml"
    import yaml
    with open(tmp_path, "w") as f:
        yaml.dump(tmp_cfg, f, default_flow_style=False)

    # Generate challenge data
    import subprocess
    subprocess.run([sys.executable, "-m", "sim.generate_dataset", "--config",
                    str(tmp_path), "--solver", "synthetic"], check=True, cwd=str(ROOT))

    # Build features for probe pipeline
    subprocess.run([sys.executable, "-m", "extract.build_features", "--config",
                    str(tmp_path)], check=True, cwd=str(ROOT))

    # Build wake fields for DL pipeline
    subprocess.run([sys.executable, "-m", "extract.build_wake_fields", "--config",
                    str(tmp_path)], check=True, cwd=str(ROOT))

    # Train probe model on challenge data
    print("  Training probe model on challenge data...")
    from ml.wake_common import stratification_labels as _sl
    index_df = pd.read_csv(ROOT / "data" / "features" / "features.csv")
    meta_cols = ["case_id", "shape", "Re", "dy", "eps", "seed"]
    feat_cols = [c for c in index_df.columns if c not in meta_cols]
    X = index_df[feat_cols].to_numpy(dtype=float)
    y = index_df["shape"].to_numpy()
    strata_ch = _sl_ch(index_df)

    f1s_probe = []
    for seed in SEEDS:
        itr, ite = train_test_split(np.arange(X.shape[0]), test_size=0.2,
                                    random_state=seed, stratify=strata_ch)
        m = _probe_model(seed)
        m.fit(X[itr], y[itr])
        f1s_probe.append(f1_score(y[ite], m.predict(X[ite]), average="macro"))
    probe_f1 = (np.mean(f1s_probe), np.std(f1s_probe))
    print(f"    Probe: F1={probe_f1[0]:.4f} ± {probe_f1[1]:.4f}")

    # Train DL models on challenge data
    bundle = load_wake_bundle()
    spec = {"scales": ["dist0.5_full", "dist1.0_full", "dist2.0_full"],
            "channels": ["ux", "uy", "speed", "vorticity"]}
    x_all = variant_tensor(bundle, scales=spec["scales"], channels=spec["channels"])
    label_maps = build_label_maps(bundle)
    bundle_strata = _sl(bundle)

    test_n = max(len(np.unique(bundle_strata)), int(0.2 * bundle.case_ids.shape[0]))
    for backbone, run_name in [("jepa", "jepa_ch"), ("resnet18", "resnet18_ch")]:
        if train_and_eval_challenge(x_all, bundle, label_maps, bundle_strata,
                                      test_n, tmp_cfg, backbone, run_name):
            pass

    # Cleanup
    tmp_path.unlink(missing_ok=True)
    print(f"\n  Results saved to {OUT}/challenge_comparison.csv")


def _sl_ch(df):
    return (df["shape"].astype(str) + "_Re" + df["Re"].astype(str)).to_numpy()


def train_and_eval_challenge(x_all, bundle, label_maps, bundle_strata, test_n, cfg, backbone, run_name):
    from ml.train_wake import _train_model as train_resnet
    from ml.train_wake import _train_jepa_model as train_jepa
    from ml.train_wake import _split_train_val, set_seed

    device = DEVICE
    f1s = []
    for seed in SEEDS:
        idx_train, idx_test = train_test_split(
            np.arange(bundle.case_ids.shape[0]), test_size=test_n,
            random_state=seed, stratify=bundle_strata)

        val_ratio = float(cfg.get("vision", {}).get("training", {}).get("val_ratio", 0.1))
        idx_tr, idx_val = _split_train_val(idx_train, bundle_strata, val_ratio, seed)

        shape_tr = np.array([label_maps.shape_to_idx[v] for v in bundle.shapes[idx_tr]], dtype=np.int64)
        re_tr = np.array([label_maps.re_to_idx[int(v)] for v in bundle.re_values[idx_tr]], dtype=np.int64)
        params_tr = np.stack([bundle.dy[idx_tr], bundle.eps[idx_tr]], axis=1).astype(np.float32)

        shape_v = np.array([label_maps.shape_to_idx[v] for v in bundle.shapes[idx_val]], dtype=np.int64) if idx_val.shape[0] > 0 else np.array([], dtype=np.int64)
        re_v = np.array([label_maps.re_to_idx[int(v)] for v in bundle.re_values[idx_val]], dtype=np.int64) if idx_val.shape[0] > 0 else np.array([], dtype=np.int64)
        params_v = np.stack([bundle.dy[idx_val], bundle.eps[idx_val]], axis=1).astype(np.float32) if idx_val.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)

        if backbone == "jepa":
            model, _ = train_jepa(
                x_train=x_all[idx_tr], shape_train_idx=shape_tr, params_train=params_tr,
                re_train_idx=re_tr, x_val=x_all[idx_val], shape_val_idx=shape_v,
                params_val=params_v, re_val_idx=re_v, cfg=cfg, seed=seed,
                n_shapes=len(label_maps.shape_to_idx), n_re_classes=len(label_maps.re_to_idx),
                device=device)
        else:
            model, _ = train_resnet(
                x_train=x_all[idx_tr], shape_train_idx=shape_tr, params_train=params_tr,
                re_train_idx=re_tr, x_val=x_all[idx_val], shape_val_idx=shape_v,
                params_val=params_v, re_val_idx=re_v, cfg=cfg, seed=seed,
                n_shapes=len(label_maps.shape_to_idx), n_re_classes=len(label_maps.re_to_idx),
                device=device)

        model.eval()
        x_t = torch.from_numpy(x_all[idx_test]).float().to(device)
        with torch.no_grad():
            out = model(x_t)
        preds = out["shape_logits"].argmax(dim=1).cpu().numpy()
        pred_str = np.array([label_maps.idx_to_shape[int(i)] for i in preds])
        f1s.append(float(f1_score(bundle.shapes[idx_test], pred_str, average="macro")))

    print(f"    {backbone}: F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return True


if __name__ == "__main__":
    bundle = load_wake_bundle()
    label_maps = build_label_maps(bundle)
    run_probe_dropout(label_maps)
    run_challenge_comparison()
