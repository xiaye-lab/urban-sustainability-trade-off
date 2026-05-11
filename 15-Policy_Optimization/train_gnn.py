"""
train_gnn.py
============
Trains GATv2 to predict ECSI and SEVI simultaneously.

Input : fishnet_final_composite.shp
        - 15 independent variables (urban morphology + accessibility + landscape)
        - ECSI: Entropy-weighted composite (LST, PM25, CO2)
        - SEVI: Entropy-TOPSIS composite (NTL, POI, RND, POPU, PRICE_PSM)

Outputs
-------
  urban_gnn_model.pth       best model weights
  target_scaler.pkl         StandardScaler on train targets
  gnn_prediction.csv        all-node predictions (original units)
  learning_curve.png        train/val loss over epochs
  baseline_comparison.csv   GATv2 vs OLS vs MLP vs graph-ablation
  vif_report.csv            multicollinearity check (from build_graph)
"""

import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from build_graph import load_graph
from gnn_model import UrbanGNN


def create_splits(num_nodes, train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)
    n_tr = int(train_ratio * num_nodes)
    n_va = int(val_ratio   * num_nodes)

    def _mask(i):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[i] = True
        return m

    return (_mask(idx[:n_tr]),
            _mask(idx[n_tr: n_tr + n_va]),
            _mask(idx[n_tr + n_va:]))


def evaluate(y_true, y_pred, label=""):
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    r2_ecsi = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_sevi = r2_score(y_true[:, 1], y_pred[:, 1])
    print(f"\n  [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"R2(ECSI)={r2_ecsi:.4f}  R2(SEVI)={r2_sevi:.4f}")
    return {"label": label, "RMSE": rmse, "MAE": mae,
            "R2_ECSI": r2_ecsi, "R2_SEVI": r2_sevi}


def plot_learning_curve(train_losses, val_losses, out="learning_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss", linewidth=1.2)
    ax.plot(val_losses,   label="Val loss",   linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised targets)")
    ax.set_title("GATv2 Learning Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load graph ───────────────────────────────────────────────────────────
    data, gdf, feature_cols, target_cols, feat_scaler = load_graph(
        shp_file="fishnet_final_composite.shp", k_sim=8
    )
    N = data.num_nodes
    F = data.x.shape[1]

    print(f"\nFeatures ({F}): {feature_cols}")
    print(f"Targets      : {target_cols}")
    print(f"Nodes        : {N:,}  |  Edges: {data.edge_index.shape[1]:,}")

    # 2. Splits ───────────────────────────────────────────────────────────────
    data.train_mask, data.val_mask, data.test_mask = create_splits(N)
    print(f"\nSplit -> train: {data.train_mask.sum():,} | "
          f"val: {data.val_mask.sum():,} | "
          f"test: {data.test_mask.sum():,}")

    # 3. Normalise targets (fit on train only) ────────────────────────────────
    y_all = data.y.numpy()
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(y_all[data.train_mask])
    data.y = torch.tensor(tgt_scaler.transform(y_all), dtype=torch.float)
    joblib.dump(tgt_scaler, "target_scaler.pkl")

    # 4. Model & optimiser ────────────────────────────────────────────────────
    model     = UrbanGNN(num_features=F, hidden_dim=64, num_targets=2, heads=4)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  patience=20, factor=0.5, min_lr=1e-6)
    loss_fn   = MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters : {total_params:,}")
    print("Hyperparameters  : lr=0.001, wd=1e-4, dropout=0.2, "
          "heads=4, hidden=64, max_epochs=800, patience=60")

    # 5. Training loop ────────────────────────────────────────────────────────
    best_val, best_state = float("inf"), None
    patience_cnt = 0
    train_losses, val_losses = [], []

    print("\n===== Training =====")
    for epoch in range(1, 801):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_weight)
        loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index, data.edge_weight)
            val_loss = loss_fn(
                val_pred[data.val_mask], data.y[data.val_mask]
            ).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 50 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:03d} | train={loss.item():.4f} | "
                  f"val={val_loss:.4f} | lr={lr_now:.2e}")

        if val_loss < best_val - 1e-6:
            best_val     = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 60:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val:.4f}")
    torch.save(model.state_dict(), "urban_gnn_model.pth")
    print("Model saved -> urban_gnn_model.pth")
    plot_learning_curve(train_losses, val_losses)

    # 6. Predictions ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = tgt_scaler.inverse_transform(
            model(data.x, data.edge_index, data.edge_weight).cpu().numpy()
        )

    y_true = y_all
    gdf["ECSI_pred"] = preds[:, 0]
    gdf["SEVI_pred"] = preds[:, 1]
    out_cols = ["OBJECTID", "X", "Y", "ECSI", "SEVI", "ECSI_pred", "SEVI_pred"]
    gdf[out_cols].to_csv("gnn_prediction.csv", index=False)
    print("Predictions saved -> gnn_prediction.csv")

    # 7. Performance metrics ──────────────────────────────────────────────────
    print("\n===== Model Performance =====")
    results = []
    for mask, label in [
        (data.train_mask.numpy(), "GATv2 Train"),
        (data.val_mask.numpy(),   "GATv2 Val"),
        (data.test_mask.numpy(),  "GATv2 Test"),
    ]:
        results.append(evaluate(y_true[mask], preds[mask], label))

    # 8. Baseline models ──────────────────────────────────────────────────────
    print("\n===== Baseline Comparison (Test set) =====")
    X_tr = data.x[data.train_mask].numpy()
    X_te = data.x[data.test_mask].numpy()
    y_tr = y_true[data.train_mask.numpy()]
    y_te = y_true[data.test_mask.numpy()]

    for bname, clf in [
        ("OLS",    LinearRegression()),
        ("MLP_sk", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                activation="relu", random_state=42,
                                early_stopping=True)),
    ]:
        clf.fit(X_tr, y_tr)
        p = clf.predict(X_te)
        results.append(evaluate(y_te, p, bname))

    # 9. Graph ablation ───────────────────────────────────────────────────────
    print("\n===== Graph Ablation (self-loops only) =====")
    self_loops = torch.stack([torch.arange(N), torch.arange(N)])
    self_w     = torch.ones(N)
    model.eval()
    with torch.no_grad():
        p_iso = tgt_scaler.inverse_transform(
            model(data.x, self_loops, self_w).cpu().numpy()
        )
    results.append(evaluate(y_te, p_iso[data.test_mask.numpy()], "Ablation (no graph)"))

    gnn_r2 = next(r for r in results if r["label"] == "GATv2 Test")
    abl_r2 = next(r for r in results if "Ablation" in r["label"])
    print(f"\n  Delta-R2 (graph contribution): "
          f"ECSI={gnn_r2['R2_ECSI'] - abl_r2['R2_ECSI']:+.4f}  "
          f"SEVI={gnn_r2['R2_SEVI'] - abl_r2['R2_SEVI']:+.4f}")

    # 10. Save comparison table ───────────────────────────────────────────────
    pd.DataFrame(results).to_csv("baseline_comparison.csv", index=False)
    print("\nSaved -> baseline_comparison.csv")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nDone. Run morans_i_analysis.py -> explainability.py -> plotting.py")
