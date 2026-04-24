import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import russian_electricity_real_kan as base_exp


ROOT = Path("/Users/kirill/Desktop/ВКР ИТМО")
OUT_DIR = ROOT / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
DEVICE = torch.device("cpu")
BATCH_SIZE = 4096
EPOCHS = 14
LR = 8e-4
WEIGHT_DECAY = 5e-5

sns.set(style="whitegrid")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric_row(model: str, y_true, pred, note: str = ""):
    return {
        "model": model,
        "RMSE": float(mean_squared_error(y_true, pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "R2": float(r2_score(y_true, pred)),
        "note": note,
    }


def compact_feature_catalog(panel: pd.DataFrame):
    features = [
        "purchase_price_rub_mwh",
        "sale_price_rub_mwh",
        "planned_consumption_mwh",
        "full_planned_consumption_mwh",
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_lag_24",
        "price_lag_48",
        "price_lag_168",
        "cons_lag_1",
        "cons_lag_24",
        "cons_lag_48",
        "cons_lag_168",
        "recipient_count",
        "total_outflow",
        "mean_outflow",
        "max_outflow",
        "daily_losses_prev_day_mwh",
        "prev_month_peak_demand_mw",
        "prev_month_energy_power_cost",
        "hour",
        "dayofweek",
        "month_num",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
    ]
    features = [f for f in features if f in panel.columns]
    feature_groups = {
        "price_history": [c for c in features if c.startswith("price_lag_") or c in {"purchase_price_rub_mwh", "sale_price_rub_mwh"}],
        "load_history": [c for c in features if c.startswith("cons_lag_") or "consumption" in c],
        "calendar": [c for c in features if c in {"hour", "dayofweek", "month_num", "is_weekend", "sin_hour", "cos_hour", "sin_dow", "cos_dow"}],
        "flows_losses": [c for c in features if c in {"recipient_count", "total_outflow", "mean_outflow", "max_outflow", "daily_losses_prev_day_mwh"}],
        "monthly_lagged": [c for c in features if c.startswith("prev_month_")],
        "region": ["region"],
    }
    feature_groups = {k: v for k, v in feature_groups.items() if v}
    return features, feature_groups


def prepare_delta_datasets(panel: pd.DataFrame):
    panel = panel.copy()
    panel["target_delta_t_plus_1"] = panel["target_price_t_plus_1"] - panel["purchase_price_rub_mwh"]

    train = panel[panel["datetime"] < "2023-11-01"].copy()
    val = panel[(panel["datetime"] >= "2023-11-01") & (panel["datetime"] < "2023-12-01")].copy()
    test = panel[(panel["datetime"] >= "2023-12-01") & (panel["datetime"] < "2024-01-01")].copy()

    numeric_features, feature_groups = compact_feature_catalog(panel)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X_train_num = num_pipe.fit_transform(train[numeric_features]).astype(np.float32)
    X_val_num = num_pipe.transform(val[numeric_features]).astype(np.float32)
    X_test_num = num_pipe.transform(test[numeric_features]).astype(np.float32)

    region_levels = sorted(train["region"].unique().tolist())
    region_to_idx = {r: i for i, r in enumerate(region_levels)}
    train_region_idx = train["region"].map(region_to_idx).astype(int).values
    val_region_idx = val["region"].map(region_to_idx).astype(int).values
    test_region_idx = test["region"].map(region_to_idx).astype(int).values

    y_train_delta = train["target_delta_t_plus_1"].astype(float).values
    y_val_delta = val["target_delta_t_plus_1"].astype(float).values
    y_test_delta = test["target_delta_t_plus_1"].astype(float).values

    delta_mean = float(y_train_delta.mean())
    delta_std = float(y_train_delta.std() + 1e-8)

    return {
        "train": train,
        "val": val,
        "test": test,
        "numeric_features": numeric_features,
        "feature_groups": feature_groups,
        "region_levels": region_levels,
        "region_to_idx": region_to_idx,
        "X_train_num": X_train_num,
        "X_val_num": X_val_num,
        "X_test_num": X_test_num,
        "train_region_idx": train_region_idx,
        "val_region_idx": val_region_idx,
        "test_region_idx": test_region_idx,
        "y_train_delta": y_train_delta.astype(np.float32),
        "y_val_delta": y_val_delta.astype(np.float32),
        "y_test_delta": y_test_delta.astype(np.float32),
        "y_train_price": train["target_price_t_plus_1"].astype(float).values.astype(np.float32),
        "y_val_price": val["target_price_t_plus_1"].astype(float).values.astype(np.float32),
        "y_test_price": test["target_price_t_plus_1"].astype(float).values.astype(np.float32),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "num_pipe": num_pipe,
    }


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 12):
        super().__init__()
        self.register_buffer("knots", torch.linspace(-3, 3, num_basis))
        self.coeffs = nn.Parameter(torch.randn(in_dim, out_dim, num_basis) * 0.03)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.knots) ** 2))
        return torch.einsum("bin,ion->bo", basis, self.coeffs) + self.bias


class KANEmbedDeltaRegressor(nn.Module):
    def __init__(self, num_dim: int, n_regions: int, embed_dim: int = 8, hidden_dim: int = 32, num_basis: int = 12):
        super().__init__()
        self.embed = nn.Embedding(n_regions, embed_dim)
        self.region_linear = nn.Linear(embed_dim, 1, bias=False)
        self.kan1 = KANLayer(num_dim, hidden_dim, num_basis=num_basis)
        self.kan2 = KANLayer(hidden_dim, 1, num_basis=num_basis)
        self.act = nn.SiLU()

    def forward(self, x_num: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embed(region_idx)
        hidden = self.act(self.kan1(x_num))
        out = self.kan2(hidden).squeeze(-1)
        out = out + self.region_linear(emb).squeeze(-1)
        return out


class HybridKANEmbedDeltaRegressor(nn.Module):
    def __init__(self, num_dim: int, n_regions: int, embed_dim: int = 8, hidden_dim: int = 32, num_basis: int = 12):
        super().__init__()
        self.embed = nn.Embedding(n_regions, embed_dim)
        self.linear = nn.Linear(num_dim + embed_dim, 1)
        self.kan1 = KANLayer(num_dim, hidden_dim, num_basis=num_basis)
        self.kan2 = KANLayer(hidden_dim, 1, num_basis=num_basis)
        self.act = nn.SiLU()

    def forward(self, x_num: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embed(region_idx)
        base = self.linear(torch.cat([x_num, emb], dim=1)).squeeze(-1)
        hidden = self.act(self.kan1(x_num))
        nonlinear = self.kan2(hidden).squeeze(-1)
        return base + nonlinear


class ResidualRidgeKANRegressor(nn.Module):
    def __init__(self, num_dim: int, n_regions: int, embed_dim: int = 8, hidden_dim: int = 32, num_basis: int = 12):
        super().__init__()
        self.embed = nn.Embedding(n_regions, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        self.kan1 = KANLayer(num_dim, hidden_dim, num_basis=num_basis)
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.SiLU()

    def forward(self, x_num: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embed(region_idx)
        hidden = self.act(self.kan1(x_num) + self.embed_proj(emb))
        return self.out(hidden).squeeze(-1)


def train_torch_model(model, X_train_num, train_region_idx, y_train, X_val_num, val_region_idx, y_val):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(train_region_idx, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_state = None
    best_val = math.inf
    patience = 4
    wait = 0
    history = []

    X_val_num_t = torch.tensor(X_val_num, dtype=torch.float32, device=DEVICE)
    val_region_t = torch.tensor(val_region_idx, dtype=torch.long, device=DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb_num, xb_region, yb in train_loader:
            xb_num = xb_num.to(DEVICE)
            xb_region = xb_region.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb_num, xb_region)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_num_t, val_region_t).cpu().numpy()
        val_rmse = float(mean_squared_error(y_val, val_pred, squared=False))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "val_rmse": val_rmse})
        print(f"{model.__class__.__name__} epoch {epoch + 1}: val_rmse={val_rmse:.4f}", flush=True)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_torch(model, X_num, region_idx):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(X_num, dtype=torch.float32, device=DEVICE),
            torch.tensor(region_idx, dtype=torch.long, device=DEVICE),
        ).cpu().numpy()
    return pred


def evaluate_stronger_models(ds):
    X_train_num = ds["X_train_num"]
    X_val_num = ds["X_val_num"]
    X_test_num = ds["X_test_num"]
    train_region_idx = ds["train_region_idx"]
    val_region_idx = ds["val_region_idx"]
    test_region_idx = ds["test_region_idx"]
    y_train_delta = ds["y_train_delta"]
    y_val_delta = ds["y_val_delta"]
    y_test_price = ds["y_test_price"]
    delta_mean = ds["delta_mean"]
    delta_std = ds["delta_std"]
    base_price_test = ds["test"]["purchase_price_rub_mwh"].values.astype(np.float32)
    n_regions = len(ds["region_levels"])

    rows = []
    preds = {}

    # Baselines on delta target.
    print("Training delta Ridge...", flush=True)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_num, y_train_delta)
    ridge_delta_test = ridge.predict(X_test_num)
    ridge_price_test = base_price_test + ridge_delta_test
    rows.append(metric_row("RidgeDelta", y_test_price, ridge_price_test, "delta_target"))
    preds["RidgeDelta"] = ridge_price_test

    print("Training delta HistGradientBoosting...", flush=True)
    hgb = HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=140,
        learning_rate=0.05,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=SEED,
    )
    hgb.fit(X_train_num, y_train_delta)
    hgb_delta_test = hgb.predict(X_test_num)
    hgb_price_test = base_price_test + hgb_delta_test
    rows.append(metric_row("HGBDelta", y_test_price, hgb_price_test, "delta_target"))
    preds["HGBDelta"] = hgb_price_test

    y_train_delta_scaled = (y_train_delta - delta_mean) / delta_std
    y_val_delta_scaled = (y_val_delta - delta_mean) / delta_std

    print("Training KAN with region embedding...", flush=True)
    kan, kan_hist = train_torch_model(
        KANEmbedDeltaRegressor(X_train_num.shape[1], n_regions, embed_dim=8, hidden_dim=32, num_basis=12),
        X_train_num,
        train_region_idx,
        y_train_delta_scaled,
        X_val_num,
        val_region_idx,
        y_val_delta_scaled,
    )
    kan_delta_test = predict_torch(kan, X_test_num, test_region_idx) * delta_std + delta_mean
    kan_price_test = base_price_test + kan_delta_test
    rows.append(metric_row("KANEmbedDelta", y_test_price, kan_price_test, "delta_target_region_embed"))
    preds["KANEmbedDelta"] = kan_price_test

    print("Training Hybrid KAN with region embedding...", flush=True)
    hybrid, hybrid_hist = train_torch_model(
        HybridKANEmbedDeltaRegressor(X_train_num.shape[1], n_regions, embed_dim=8, hidden_dim=32, num_basis=12),
        X_train_num,
        train_region_idx,
        y_train_delta_scaled,
        X_val_num,
        val_region_idx,
        y_val_delta_scaled,
    )
    hybrid_delta_test = predict_torch(hybrid, X_test_num, test_region_idx) * delta_std + delta_mean
    hybrid_price_test = base_price_test + hybrid_delta_test
    rows.append(metric_row("HybridKANEmbedDelta", y_test_price, hybrid_price_test, "delta_target_region_embed"))
    preds["HybridKANEmbedDelta"] = hybrid_price_test

    print("Training residual Ridge + KAN...", flush=True)
    ridge_delta_train = ridge.predict(X_train_num)
    ridge_delta_val = ridge.predict(X_val_num)
    train_resid = y_train_delta - ridge_delta_train
    val_resid = y_val_delta - ridge_delta_val
    resid_mean = float(train_resid.mean())
    resid_std = float(train_resid.std() + 1e-8)
    train_resid_scaled = (train_resid - resid_mean) / resid_std
    val_resid_scaled = (val_resid - resid_mean) / resid_std

    residual_kan, residual_hist = train_torch_model(
        ResidualRidgeKANRegressor(X_train_num.shape[1], n_regions, embed_dim=8, hidden_dim=32, num_basis=12),
        X_train_num,
        train_region_idx,
        train_resid_scaled,
        X_val_num,
        val_region_idx,
        val_resid_scaled,
    )
    resid_delta_test = predict_torch(residual_kan, X_test_num, test_region_idx) * resid_std + resid_mean
    ridge_kan_delta_test = ridge_delta_test + resid_delta_test
    ridge_kan_price_test = base_price_test + ridge_kan_delta_test
    rows.append(metric_row("ResidualRidgeKAN", y_test_price, ridge_kan_price_test, "delta_target_residual"))
    preds["ResidualRidgeKAN"] = ridge_kan_price_test

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    history = pd.concat(
        [
            kan_hist.assign(model="KANEmbedDelta"),
            hybrid_hist.assign(model="HybridKANEmbedDelta"),
            residual_hist.assign(model="ResidualRidgeKAN"),
        ],
        ignore_index=True,
    )
    models = {
        "KANEmbedDelta": kan,
        "HybridKANEmbedDelta": hybrid,
        "ResidualRidgeKAN": residual_kan,
        "RidgeDelta": ridge,
        "HGBDelta": hgb,
    }
    return metrics, preds, models, history


def save_outputs(ds, metrics, preds, history):
    metrics.to_csv(OUT_DIR / "russian_electricity_stronger_kan_metrics.csv", index=False)
    history.to_csv(OUT_DIR / "russian_electricity_stronger_kan_train_history.csv", index=False)

    best_name = metrics.iloc[0]["model"]
    pred_df = ds["test"][["datetime", "region", "hour", "purchase_price_rub_mwh", "target_price_t_plus_1"]].copy()
    pred_df["prediction"] = preds[best_name]
    pred_df["residual"] = pred_df["target_price_t_plus_1"] - pred_df["prediction"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    pred_df["squared_error"] = pred_df["residual"] ** 2
    pred_df["model"] = best_name
    pred_df.to_csv(OUT_DIR / "russian_electricity_stronger_kan_predictions.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.barplot(data=metrics, x="model", y="RMSE", color="#6c8ebf")
    plt.xticks(rotation=20)
    plt.title("Усиленные KAN-конфигурации и baseline на delta target")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_stronger_kan_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=history, x="epoch", y="val_rmse", hue="model", marker="o")
    plt.title("Validation RMSE при обучении усиленных KAN-моделей")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_stronger_kan_training.png", dpi=160)
    plt.close()

    err_by_region = pred_df.groupby("region").agg(
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda s: float(np.sqrt(s.mean()))),
    ).reset_index().sort_values("RMSE", ascending=False)
    err_by_region.to_csv(OUT_DIR / "russian_electricity_stronger_kan_error_by_region.csv", index=False)

    summary_lines = [
        "## Усиленный KAN-эксперимент",
        "",
        "- Целевая переменная задана как `delta price = price(t+1) - price(t)`.",
        "- Добавлены `region embedding` и компактный набор признаков.",
        "- Добавлена residual-схема: `Ridge` прогнозирует базовую часть, `KAN` доучивает остаток.",
        "",
        "## Метрики",
        "",
        metrics.to_string(index=False),
        "",
        f"Лучшая модель: `{best_name}` с RMSE `{metrics.iloc[0]['RMSE']:.3f}`.",
    ]
    (OUT_DIR / "russian_electricity_stronger_kan_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (OUT_DIR / "russian_electricity_stronger_kan_summary.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "best_rmse": float(metrics.iloc[0]["RMSE"]),
                "best_mae": float(metrics.iloc[0]["MAE"]),
                "rows_test": int(len(ds["test"])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    set_seed(SEED)
    panel, _ = base_exp.load_or_build_panel()
    ds = prepare_delta_datasets(panel)
    metrics, preds, _, history = evaluate_stronger_models(ds)
    save_outputs(ds, metrics, preds, history)
    print(metrics.to_string(index=False))
    print("Saved metrics to:", OUT_DIR / "russian_electricity_stronger_kan_metrics.csv")


if __name__ == "__main__":
    main()
