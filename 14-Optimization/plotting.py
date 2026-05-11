"""
plotting.py
===========
SHAP dependence plots for ECSI and SEVI — one scatter per feature per target.
Run AFTER explainability.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_shap_dependence(target_features=None):
    print("\nGenerating SHAP dependence plots ...")

    # 1. Load artefacts ──────────────────────────────────────────────────────
    try:
        shap_ecsi     = np.load("shap_values_ecsi.npy")
        shap_sevi     = np.load("shap_values_sevi.npy")
        X_explain     = np.load("shap_X_explain.npy")
        feature_names = pd.read_csv("shap_feature_names.csv")["Feature"].tolist()

        try:
            df_true = pd.read_csv("gnn_prediction.csv")
        except FileNotFoundError:
            df_true = pd.read_csv("nodes_with_ECSI_SEVI.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}\nRun explainability.py first.")
        return

    # 2. Shape cleaning ──────────────────────────────────────────────────────
    def clean_shape(arr, ref_X):
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.shape[0] != ref_X.shape[0] and arr.shape[1] == ref_X.shape[0]:
            arr = arr.T
        return arr

    shap_ecsi = clean_shape(shap_ecsi, X_explain)
    shap_sevi = clean_shape(shap_sevi, X_explain)

    # 3. Helper: un-scale feature values ────────────────────────────────────
    def get_true_values(feat_name, scaled_vals):
        if feat_name in df_true.columns:
            mu    = df_true[feat_name].mean()
            sigma = df_true[feat_name].std()
            return scaled_vals * sigma + mu
        return scaled_vals

    if target_features is None:
        target_features = feature_names

    # 4. Plot function ────────────────────────────────────────────────────────
    def create_plot(feat_name, shap_values, target_label, color_label):
        idx    = feature_names.index(feat_name)
        x_data = get_true_values(feat_name, X_explain[:, idx])
        c_data = x_data                      # colour by own value

        plt.figure(figsize=(10, 7))
        sc = plt.scatter(
            x_data, shap_values[:, idx],
            c=c_data, cmap="coolwarm",
            s=10, alpha=0.8, edgecolor="none",
        )
        cbar = plt.colorbar(sc)
        cbar.set_label(f"{feat_name} value", fontsize=11)

        plt.xlabel(f"{feat_name} (true value)", fontsize=13)
        plt.ylabel(
            f"SHAP value for {feat_name}\n(impact on {target_label})", fontsize=13
        )
        plt.title(f"Dependence plot: {feat_name} → {target_label}", fontsize=15)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        fname = f"shap_dep_{feat_name}_{target_label}.png"
        plt.savefig(fname, dpi=300)
        print(f"  Saved → {fname}")
        plt.close()

    # 5. Generate plots ───────────────────────────────────────────────────────
    for feat in target_features:
        if feat not in feature_names:
            print(f"  Skipping '{feat}' — not in feature list.")
            continue
        create_plot(feat, shap_ecsi, "ECSI", feat)
        create_plot(feat, shap_sevi, "SEVI", feat)


if __name__ == "__main__":
    plot_shap_dependence(target_features=None)
