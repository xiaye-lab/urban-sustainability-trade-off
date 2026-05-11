"""
plot_beeswarm.py
================
Side-by-side SHAP beeswarm plots for ECSI and SEVI.
Run AFTER explainability.py.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_beeswarm_comparison():
    print("Generating SHAP beeswarm comparison plot ...")

    # 1. Load SHAP artefacts ─────────────────────────────────────────────────
    try:
        shap_ecsi    = np.load("shap_values_ecsi.npy")
        shap_sevi    = np.load("shap_values_sevi.npy")
        X_explain    = np.load("shap_X_explain.npy")
        feature_names = pd.read_csv("shap_feature_names.csv")["Feature"].tolist()
    except FileNotFoundError as e:
        print(f"Error: {e}\nRun explainability.py first.")
        return

    # 2. Ensure (N, F) shape ─────────────────────────────────────────────────
    def clean_shape(arr, ref_X):
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.shape[0] != ref_X.shape[0] and arr.shape[1] == ref_X.shape[0]:
            arr = arr.T
        return arr

    shap_ecsi = clean_shape(shap_ecsi, X_explain)
    shap_sevi = clean_shape(shap_sevi, X_explain)

    # 3. Side-by-side plot ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(1, 2, 1)
    plt.sca(ax1)
    shap.summary_plot(
        shap_ecsi, X_explain, feature_names=feature_names,
        show=False, plot_type="dot", cmap="coolwarm", sort=True,
    )
    ax1.set_title("Impact on ECSI (Environmental Condition Stress Index)", fontsize=14)
    ax1.set_xlabel("SHAP value (impact on ECSI)", fontsize=12)

    ax2 = fig.add_subplot(1, 2, 2)
    plt.sca(ax2)
    shap.summary_plot(
        shap_sevi, X_explain, feature_names=feature_names,
        show=False, plot_type="dot", cmap="coolwarm", sort=True,
    )
    ax2.set_title("Impact on SEVI (Socioeconomic Vitality Index)", fontsize=14)
    ax2.set_xlabel("SHAP value (impact on SEVI)", fontsize=12)

    plt.tight_layout()
    plt.savefig("shap_beeswarm_comparison.png", dpi=300)
    print("Saved → shap_beeswarm_comparison.png")
    plt.close()


if __name__ == "__main__":
    plot_beeswarm_comparison()
