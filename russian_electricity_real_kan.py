import os

# Reduce OpenMP-related issues before importing numpy/torch/sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import glob
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/Users/kirill/Desktop/ВКР ИТМО")
DATA_DIR = ROOT / "Данные электроэнергии"
OUT_DIR = ROOT / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
DEVICE = torch.device("cpu")
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

sns.set(style="whitegrid")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def region_name(path: Path, suffix: str) -> str:
    name = path.name
    return name[len("region_") : -len(suffix)]


def to_num(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    value = str(value).strip().replace(" ", "").replace("\xa0", "")
    if value in {"", "-", "—"}:
        return np.nan
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return np.nan


def discover_region_files():
    hourly_dir = DATA_DIR / "торги и перетоки почасовые"
    daily_dir = DATA_DIR / "посуточные"
    monthly_dir = DATA_DIR / "ставки, стоимость, отчет - месяц"

    hourly = {region_name(Path(p), "_data.csv"): Path(p) for p in glob.glob(str(hourly_dir / "region_*_data.csv"))}
    daily = {region_name(Path(p), "_day.csv"): Path(p) for p in glob.glob(str(daily_dir / "region_*_day.csv"))}
    monthly = {region_name(Path(p), "_mon.csv"): Path(p) for p in glob.glob(str(monthly_dir / "region_*_mon.csv"))}

    common_regions = sorted(set(hourly) & set(daily) & set(monthly))
    return hourly, daily, monthly, common_regions


def build_leakage_safe_panel():
    hourly, daily, monthly, regions = discover_region_files()

    monthly_cols = {
        'Стоимость услуги по оперативно-диспетчерскому управлению в электроэнергетике, оказанной АО "СО ЕЭС", приходящаяся на ГТП потребления с признаком гарантирующего поставщика': "so_service_cost",
        "Стоимость комплексной услуги АО «ЦФР», приходящаяся на ГТП потребления с признаком гарантирующего поставщика": "cfr_service_cost",
        "P_субъект_РФ, Совокупная нерегулируемая часть объема фактического пикового потребления в субъекте Российской Федерации, МВт": "peak_demand_mw",
        "Стоимость электрической энергии и мощности для целей определения ставки тарифа на услуги по передаче электрической энергии, используемой для целей определения расходов на оплату нормативных потерь электрической энергии при ее передаче по электрическим сетям единой национальной (общероссийской) электрической сети": "energy_power_cost",
        "Объем потерь электрической энергии в электрических сетях единой национальной (общероссийской) электрической сети ": "grid_losses_volume",
        "Ставка тарифа на услуги по передаче электрической энергии, используемая для целей определения расходов на оплату нормативных потерь электрической энергии при ее передаче по электрическим сетям единой национальной (общероссийской) электрической сети": "grid_loss_tariff_rate",
    }

    base_rename = {
        "Плановый объём потребления, МВт.ч._first": "planned_consumption_mwh",
        "Плановый объём экспорта, МВт.ч._first": "planned_export_mwh",
        "Плановый объём импорта, МВт.ч._first": "planned_import_mwh",
        "Средневзвешенная цена на покупку электроэнергии, руб./МВт.ч._first": "purchase_price_rub_mwh",
        "Средневзвешенная цена на продажу электроэнергии, руб./МВт.ч._first": "sale_price_rub_mwh",
        "Полный плановый объем потребления, МВт.ч._first": "full_planned_consumption_mwh",
        "Субъект РФ - Получатель_nunique": "recipient_count",
        "Объём перетока_sum": "total_outflow",
        "Объём перетока_mean": "mean_outflow",
        "Объём перетока_max": "max_outflow",
    }

    hourly_keep = [
        "ГЭС",
        "АЭС",
        "ТЭС",
        "СЭС",
        "ВЭС",
        "прочие ВИЭ",
        "ГЭС.1",
        "АЭС.1",
        "ТЭС.1",
        "СЭС.1",
        "ВЭС.1",
        "прочие ВИЭ.1",
        "ГЭС.2",
        "АЭС.2",
        "ТЭС.2",
        "СЭС.2",
        "ВЭС.2",
        "прочие ВИЭ.2",
        "ГЭС.3",
        "АЭС.3",
        "ТЭС.3",
        "СЭС.3",
        "ВЭС.3",
        "прочие ВИЭ.3",
        "Плановый объём потребления, МВт.ч.",
        "Плановый объём экспорта, МВт.ч.",
        "Плановый объём импорта, МВт.ч.",
        "Средневзвешенная цена на покупку электроэнергии, руб./МВт.ч.",
        "Средневзвешенная цена на продажу электроэнергии, руб./МВт.ч.",
        "Полный плановый объем потребления, МВт.ч.",
    ]

    frames = []
    daily_min, daily_max, month_min, month_max = [], [], [], []

    for region in regions:
        h = pd.read_csv(hourly[region])
        h["Дата"] = pd.to_datetime(h["Дата"])
        agg_spec = {col: "first" for col in hourly_keep if col in h.columns}
        agg_spec["Субъект РФ - Получатель"] = "nunique"
        agg_spec["Объём перетока"] = ["sum", "mean", "max"]
        g = h.groupby(["Субъект РФ", "Дата", "Час"]).agg(agg_spec)
        g.columns = ["_".join([x for x in col if x]).strip("_") for col in g.columns]
        g = g.reset_index().rename(columns=base_rename)

        d = pd.read_csv(daily[region])
        d["Дата"] = pd.to_datetime(d["Дата"])
        d = d[["Дата", "Объем потерь, МВтЧ."]].rename(columns={"Объем потерь, МВтЧ.": "daily_losses_mwh_raw"})
        d = d.sort_values("Дата")
        d["daily_losses_prev_day_mwh"] = d["daily_losses_mwh_raw"].shift(1)
        d = d[["Дата", "daily_losses_prev_day_mwh"]]
        daily_min.append(d["Дата"].min())
        daily_max.append(d["Дата"].max())

        m = pd.read_csv(monthly[region])
        for src, dst in monthly_cols.items():
            if src in m.columns:
                m[dst] = m[src].map(to_num)
        m["month"] = pd.to_datetime(m["Дата_x"], dayfirst=True, errors="coerce").dt.to_period("M").dt.to_timestamp()
        month_min.append(m["month"].min())
        month_max.append(m["month"].max())
        month_features = ["so_service_cost", "cfr_service_cost", "peak_demand_mw", "energy_power_cost", "grid_losses_volume", "grid_loss_tariff_rate"]
        m_agg = m.groupby("month")[month_features].agg(
            {
                "so_service_cost": "sum",
                "cfr_service_cost": "sum",
                "peak_demand_mw": "max",
                "energy_power_cost": "max",
                "grid_losses_volume": "max",
                "grid_loss_tariff_rate": "max",
            }
        ).reset_index()
        m_agg = m_agg.sort_values("month")
        for col in month_features:
            m_agg[f"prev_month_{col}"] = m_agg[col].shift(1)
        m_agg = m_agg[["month"] + [f"prev_month_{c}" for c in month_features]]

        g["month"] = g["Дата"].dt.to_period("M").dt.to_timestamp()
        out = g.merge(d, on="Дата", how="left").merge(m_agg, on="month", how="left")
        out["region"] = region
        frames.append(out)

    panel = pd.concat(frames, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["Дата"]) + pd.to_timedelta(panel["Час"], unit="h")

    common_start = max(min(daily_min), min(month_min))
    common_end = min(max(daily_max), max(month_max) + pd.offsets.MonthEnd(0))
    panel = panel[(panel["Дата"] >= common_start) & (panel["Дата"] <= common_end)].copy()
    panel = panel.sort_values(["region", "datetime"]).reset_index(drop=True)

    for lag in [1, 2, 3, 24, 25, 48, 168]:
        panel[f"price_lag_{lag}"] = panel.groupby("region")["purchase_price_rub_mwh"].shift(lag)
        panel[f"cons_lag_{lag}"] = panel.groupby("region")["planned_consumption_mwh"].shift(lag)

    panel["target_price_t_plus_1"] = panel.groupby("region")["purchase_price_rub_mwh"].shift(-1)
    panel["hour"] = panel["datetime"].dt.hour
    panel["dayofweek"] = panel["datetime"].dt.dayofweek
    panel["month_num"] = panel["datetime"].dt.month
    panel["is_weekend"] = (panel["dayofweek"] >= 5).astype(int)
    panel["sin_hour"] = np.sin(2 * np.pi * panel["hour"] / 24)
    panel["cos_hour"] = np.cos(2 * np.pi * panel["hour"] / 24)
    panel["sin_dow"] = np.sin(2 * np.pi * panel["dayofweek"] / 7)
    panel["cos_dow"] = np.cos(2 * np.pi * panel["dayofweek"] / 7)

    panel = panel.dropna(subset=["target_price_t_plus_1", "price_lag_168", "cons_lag_168"]).copy()
    panel = panel.drop(columns=[col for col in panel.columns if panel[col].isna().mean() == 1.0])

    meta = {
        "regions_total": len(regions),
        "regions_in_panel": int(panel["region"].nunique()),
        "common_window_start": str(common_start.date()),
        "common_window_end": str(common_end.date()),
        "panel_rows": int(len(panel)),
        "panel_columns": int(panel.shape[1]),
        "datetime_min": str(panel["datetime"].min()),
        "datetime_max": str(panel["datetime"].max()),
    }
    return panel, meta


def load_or_build_panel():
    panel_path = OUT_DIR / "russian_electricity_panel_dataset_leakage_safe.csv"
    summary_path = OUT_DIR / "russian_electricity_real_kan_summary.json"
    if panel_path.exists():
        panel = pd.read_csv(panel_path, parse_dates=["Дата", "datetime", "month"])
        meta = {
            "regions_total": int(panel["region"].nunique()),
            "regions_in_panel": int(panel["region"].nunique()),
            "common_window_start": str(panel["Дата"].min().date()),
            "common_window_end": str(panel["Дата"].max().date()),
            "panel_rows": int(len(panel)),
            "panel_columns": int(panel.shape[1]),
            "datetime_min": str(panel["datetime"].min()),
            "datetime_max": str(panel["datetime"].max()),
        }
        if summary_path.exists():
            try:
                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                meta["regions_total"] = prev.get("regions_total", meta["regions_total"])
                meta["common_window_start"] = prev.get("common_window_start", meta["common_window_start"])
                meta["common_window_end"] = prev.get("common_window_end", meta["common_window_end"])
            except Exception:
                pass
        return panel, meta
    return build_leakage_safe_panel()


def feature_catalog(panel: pd.DataFrame):
    sparse_generation_cols = [
        "ГЭС_first",
        "АЭС_first",
        "СЭС_first",
        "ВЭС_first",
        "ГЭС.1_first",
        "АЭС.1_first",
        "СЭС.1_first",
        "ВЭС.1_first",
        "ГЭС.2_first",
        "АЭС.2_first",
        "СЭС.2_first",
        "ВЭС.2_first",
        "ГЭС.3_first",
        "АЭС.3_first",
        "СЭС.3_first",
        "ВЭС.3_first",
    ]
    dense_generation_cols = ["ТЭС_first", "ТЭС.1_first", "ТЭС.2_first", "ТЭС.3_first"]

    numeric_features = [
        "Час",
        "planned_consumption_mwh",
        "planned_export_mwh",
        "planned_import_mwh",
        "full_planned_consumption_mwh",
        "recipient_count",
        "total_outflow",
        "mean_outflow",
        "max_outflow",
        "daily_losses_prev_day_mwh",
        "prev_month_so_service_cost",
        "prev_month_cfr_service_cost",
        "prev_month_peak_demand_mw",
        "prev_month_energy_power_cost",
        "prev_month_grid_losses_volume",
        "prev_month_grid_loss_tariff_rate",
        "purchase_price_rub_mwh",
        "sale_price_rub_mwh",
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_lag_24",
        "price_lag_25",
        "price_lag_48",
        "price_lag_168",
        "cons_lag_1",
        "cons_lag_2",
        "cons_lag_3",
        "cons_lag_24",
        "cons_lag_25",
        "cons_lag_48",
        "cons_lag_168",
        "hour",
        "dayofweek",
        "month_num",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
    ] + sparse_generation_cols + dense_generation_cols

    numeric_features = [col for col in numeric_features if col in panel.columns]
    feature_groups = {
        "price_history": [c for c in numeric_features if c.startswith("price_lag_") or c == "purchase_price_rub_mwh" or c == "sale_price_rub_mwh"],
        "load_history": [c for c in numeric_features if c.startswith("cons_lag_") or "consumption" in c],
        "calendar": [c for c in numeric_features if c in {"Час", "hour", "dayofweek", "month_num", "is_weekend", "sin_hour", "cos_hour", "sin_dow", "cos_dow"}],
        "flows_losses": [c for c in numeric_features if c in {"recipient_count", "total_outflow", "mean_outflow", "max_outflow", "daily_losses_prev_day_mwh"}],
        "monthly_lagged": [c for c in numeric_features if c.startswith("prev_month_")],
        "generation_mix": [c for c in numeric_features if "ЭС" in c or "ВИЭ" in c],
        "region": ["region"],
    }
    feature_groups = {k: v for k, v in feature_groups.items() if v}
    return numeric_features, sparse_generation_cols + dense_generation_cols, feature_groups


def prepare_datasets(panel: pd.DataFrame):
    train = panel[panel["datetime"] < "2023-11-01"].copy()
    val = panel[(panel["datetime"] >= "2023-11-01") & (panel["datetime"] < "2023-12-01")].copy()
    test = panel[(panel["datetime"] >= "2023-12-01") & (panel["datetime"] < "2024-01-01")].copy()

    numeric_features, generation_cols, feature_groups = feature_catalog(panel)
    for col in generation_cols:
        if col in panel.columns:
            train[col] = train[col].fillna(0.0)
            val[col] = val[col].fillna(0.0)
            test[col] = test[col].fillna(0.0)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_train_num = num_pipe.fit_transform(train[numeric_features])
    X_val_num = num_pipe.transform(val[numeric_features])
    X_test_num = num_pipe.transform(test[numeric_features])

    X_train_cat = cat_pipe.fit_transform(train[["region"]])
    X_val_cat = cat_pipe.transform(val[["region"]])
    X_test_cat = cat_pipe.transform(test[["region"]])

    y_train = train["target_price_t_plus_1"].astype(float).values
    y_val = val["target_price_t_plus_1"].astype(float).values
    y_test = test["target_price_t_plus_1"].astype(float).values
    target_mean = float(y_train.mean())
    target_std = float(y_train.std() + 1e-8)

    return {
        "train": train,
        "val": val,
        "test": test,
        "numeric_features": numeric_features,
        "region_levels": list(cat_pipe.categories_[0]),
        "feature_groups": feature_groups,
        "X_train_num": X_train_num.astype(np.float32),
        "X_val_num": X_val_num.astype(np.float32),
        "X_test_num": X_test_num.astype(np.float32),
        "X_train_cat": X_train_cat.astype(np.float32),
        "X_val_cat": X_val_cat.astype(np.float32),
        "X_test_cat": X_test_cat.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
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

    def phi(self, x_grid: torch.Tensor) -> torch.Tensor:
        grid = x_grid.view(-1, 1, 1)
        basis = torch.exp(-((grid - self.knots.view(1, 1, -1)) ** 2))
        coeffs = self.coeffs.unsqueeze(0)
        return (basis.unsqueeze(2) * coeffs).sum(-1)


class KANRegressor(nn.Module):
    def __init__(self, num_dim: int, cat_dim: int, hidden_dim: int = 24, num_basis: int = 12):
        super().__init__()
        self.kan1 = KANLayer(num_dim, hidden_dim, num_basis=num_basis)
        self.kan2 = KANLayer(hidden_dim, 1, num_basis=num_basis)
        self.cat_linear = nn.Linear(cat_dim, 1, bias=False) if cat_dim else None
        self.act = nn.SiLU()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.kan1(x_num))
        out = self.kan2(hidden)
        if self.cat_linear is not None:
            out = out + self.cat_linear(x_cat)
        return out.squeeze(-1)


class HybridKANRegressor(nn.Module):
    def __init__(self, num_dim: int, cat_dim: int, hidden_dim: int = 24, num_basis: int = 12):
        super().__init__()
        self.linear = nn.Linear(num_dim + cat_dim, 1)
        self.kan1 = KANLayer(num_dim, hidden_dim, num_basis=num_basis)
        self.kan2 = KANLayer(hidden_dim, 1, num_basis=num_basis)
        self.act = nn.SiLU()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        base = self.linear(torch.cat([x_num, x_cat], dim=1)).squeeze(-1)
        hidden = self.act(self.kan1(x_num))
        nonlinear = self.kan2(hidden).squeeze(-1)
        return base + nonlinear


def metric_row(model: str, y_true, pred, note: str = ""):
    return {
        "model": model,
        "RMSE": float(mean_squared_error(y_true, pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "R2": float(r2_score(y_true, pred)),
        "note": note,
    }


def train_torch_model(model, X_train_num, X_train_cat, y_train, X_val_num, X_val_cat, y_val):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_state = None
    best_val = math.inf
    history = []
    patience = 4
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(DEVICE)
            xb_cat = xb_cat.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(
                torch.tensor(X_val_num, dtype=torch.float32, device=DEVICE),
                torch.tensor(X_val_cat, dtype=torch.float32, device=DEVICE),
            ).cpu().numpy()
        val_rmse = float(mean_squared_error(y_val, val_pred, squared=False))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(train_losses)), "val_rmse": val_rmse})
        print(f"{model.__class__.__name__} epoch {epoch + 1}: val_rmse={val_rmse:.3f}", flush=True)
        if val_rmse < best_val:
            best_val = val_rmse
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_torch(model, X_num, X_cat):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(X_num, dtype=torch.float32, device=DEVICE),
            torch.tensor(X_cat, dtype=torch.float32, device=DEVICE),
        ).cpu().numpy()
    return pred


def evaluate_models(ds):
    train_num = ds["X_train_num"]
    val_num = ds["X_val_num"]
    test_num = ds["X_test_num"]
    train_cat = ds["X_train_cat"]
    val_cat = ds["X_val_cat"]
    test_cat = ds["X_test_cat"]
    y_train = ds["y_train"]
    y_val = ds["y_val"]
    y_test = ds["y_test"]
    y_mean = ds["target_mean"]
    y_std = ds["target_std"]
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    X_train_full = np.hstack([train_num, train_cat])
    X_val_full = np.hstack([val_num, val_cat])
    X_test_full = np.hstack([test_num, test_cat])
    X_stack = np.vstack([X_train_full, X_val_full])
    y_stack = np.hstack([y_train, y_val])

    rows = []
    preds = {}

    test_frame = ds["test"]
    print("Evaluating naive baselines...", flush=True)
    rows.append(metric_row("NaiveCurrentPrice", y_test, test_frame["purchase_price_rub_mwh"].values, "forecast_t+1"))
    preds["NaiveCurrentPrice"] = test_frame["purchase_price_rub_mwh"].values

    seasonal_day = test_frame["price_lag_23"].values if "price_lag_23" in test_frame.columns else test_frame["price_lag_24"].values
    rows.append(metric_row("SeasonalNaiveDay", y_test, seasonal_day, "forecast_t+1"))
    preds["SeasonalNaiveDay"] = seasonal_day

    if "price_lag_167" in test_frame.columns:
        seasonal_week = test_frame["price_lag_167"].values
    else:
        seasonal_week = test_frame["price_lag_168"].values
    rows.append(metric_row("SeasonalNaiveWeek", y_test, seasonal_week, "forecast_t+1"))
    preds["SeasonalNaiveWeek"] = seasonal_week

    print("Training linear models...", flush=True)
    linear = LinearRegression().fit(X_stack, y_stack)
    linear_pred = linear.predict(X_test_full)
    rows.append(metric_row("LinearRegression", y_test, linear_pred, "full"))
    preds["LinearRegression"] = linear_pred

    ridge = Ridge(alpha=1.0).fit(X_stack, y_stack)
    ridge_pred = ridge.predict(X_test_full)
    rows.append(metric_row("Ridge", y_test, ridge_pred, "full"))
    preds["Ridge"] = ridge_pred

    print("Training HistGradientBoosting on subsample...", flush=True)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X_stack), size=min(30000, len(X_stack)), replace=False)
    hgb = HistGradientBoostingRegressor(max_depth=5, max_iter=100, learning_rate=0.06, min_samples_leaf=60, l2_regularization=0.1, random_state=SEED)
    hgb.fit(X_stack[sample_idx], y_stack[sample_idx])
    hgb_pred = hgb.predict(X_test_full)
    rows.append(metric_row("HistGradientBoosting", y_test, hgb_pred, "full"))
    preds["HistGradientBoosting"] = hgb_pred

    print("Training true KAN...", flush=True)
    kan, kan_hist = train_torch_model(
        KANRegressor(num_dim=train_num.shape[1], cat_dim=train_cat.shape[1], hidden_dim=16, num_basis=8),
        train_num,
        train_cat,
        y_train_scaled,
        val_num,
        val_cat,
        y_val_scaled,
    )
    kan_pred = predict_torch(kan, test_num, test_cat) * y_std + y_mean
    rows.append(metric_row("KAN", y_test, kan_pred, "torch_true_kan"))
    preds["KAN"] = kan_pred

    print("Training hybrid true KAN...", flush=True)
    hybrid, hybrid_hist = train_torch_model(
        HybridKANRegressor(num_dim=train_num.shape[1], cat_dim=train_cat.shape[1], hidden_dim=16, num_basis=8),
        train_num,
        train_cat,
        y_train_scaled,
        val_num,
        val_cat,
        y_val_scaled,
    )
    hybrid_pred = predict_torch(hybrid, test_num, test_cat) * y_std + y_mean
    rows.append(metric_row("HybridKAN", y_test, hybrid_pred, "torch_true_kan"))
    preds["HybridKAN"] = hybrid_pred

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    train_history = pd.concat(
        [
            kan_hist.assign(model="KAN"),
            hybrid_hist.assign(model="HybridKAN"),
        ],
        ignore_index=True,
    )
    return metrics, preds, {"KAN": kan, "HybridKAN": hybrid, "LinearRegression": linear, "Ridge": ridge, "HistGradientBoosting": hgb}, train_history


def permutation_importance_groups(model, ds, group_map):
    X_num = ds["X_test_num"].copy()
    X_cat = ds["X_test_cat"].copy()
    y = ds["y_test"]

    baseline = predict_torch(model, X_num, X_cat) * ds["target_std"] + ds["target_mean"]
    baseline_rmse = float(mean_squared_error(y, baseline, squared=False))

    feature_to_idx = {name: i for i, name in enumerate(ds["numeric_features"])}
    region_group = list(range(X_cat.shape[1]))
    rows = []

    rng = np.random.default_rng(SEED)
    for group_name, cols in group_map.items():
        X_num_perm = X_num.copy()
        X_cat_perm = X_cat.copy()
        if group_name == "region":
            idx = rng.permutation(len(X_cat_perm))
            X_cat_perm = X_cat_perm[idx]
        else:
            col_idx = [feature_to_idx[c] for c in cols if c in feature_to_idx]
            if not col_idx:
                continue
            idx = rng.permutation(len(X_num_perm))
            X_num_perm[:, col_idx] = X_num_perm[idx][:, col_idx]
        pred = predict_torch(model, X_num_perm, X_cat_perm) * ds["target_std"] + ds["target_mean"]
        rmse = float(mean_squared_error(y, pred, squared=False))
        rows.append({"group": group_name, "baseline_rmse": baseline_rmse, "rmse_after_permutation": rmse, "rmse_delta": rmse - baseline_rmse})
    return pd.DataFrame(rows).sort_values("rmse_delta", ascending=False).reset_index(drop=True)


def feature_importance_kan(model, ds):
    x = torch.tensor(ds["X_test_num"], dtype=torch.float32)
    coeffs = model.kan1.coeffs.detach().cpu()
    knots = model.kan1.knots.detach().cpu()
    basis = torch.exp(-((x.unsqueeze(-1) - knots.view(1, 1, -1)) ** 2))
    contrib = torch.einsum("bin,ion->bio", basis, coeffs).abs().mean(0).sum(-1).numpy()
    return pd.DataFrame({"feature": ds["numeric_features"], "importance": contrib}).sort_values("importance", ascending=False).reset_index(drop=True)


def save_phi_plots(model, ds, prefix: str):
    importance_df = feature_importance_kan(model, ds)
    top_features = importance_df.head(6)["feature"].tolist()
    feature_to_idx = {name: i for i, name in enumerate(ds["numeric_features"])}
    x_grid = torch.linspace(-3, 3, 240)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    records = []
    for ax, feature in zip(axes.flat, top_features):
        i = feature_to_idx[feature]
        coeff = model.kan1.coeffs.detach().cpu()[i]
        hidden_idx = int(torch.norm(coeff, dim=1).argmax().item())
        y_grid = model.kan1.phi(x_grid)[:, i, hidden_idx].detach().numpy()
        ax.plot(x_grid.numpy(), y_grid, color="#1d3557")
        ax.set_title(f"{feature} -> h{hidden_idx}")
        ax.set_xlabel("scaled x")
        ax.set_ylabel("phi(x)")
        for xv, yv in zip(x_grid.numpy(), y_grid):
            records.append({"feature": feature, "hidden_unit": hidden_idx, "scaled_x": float(xv), "phi_value": float(yv)})

    for ax in axes.flat[len(top_features) :]:
        ax.axis("off")

    plt.suptitle(f"{prefix}: phi-функции первого KAN-слоя", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix.lower()}_phi_functions.png", dpi=160)
    plt.close()

    pd.DataFrame(records).to_csv(OUT_DIR / f"{prefix.lower()}_phi_functions.csv", index=False)
    return importance_df


def save_diagnostics(ds, preds, metrics, models, train_history):
    y_test = ds["y_test"]
    kan_family = metrics[metrics["model"].isin(["KAN", "HybridKAN"])].sort_values("RMSE").reset_index(drop=True)
    best_model_name = kan_family.iloc[0]["model"] if not kan_family.empty else metrics.iloc[0]["model"]
    best_pred = preds[best_model_name]

    pred_df = ds["test"][["datetime", "region", "hour", "purchase_price_rub_mwh", "target_price_t_plus_1"]].copy()
    pred_df["prediction"] = best_pred
    pred_df["residual"] = pred_df["target_price_t_plus_1"] - pred_df["prediction"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    pred_df["squared_error"] = pred_df["residual"] ** 2
    pred_df["model"] = best_model_name
    pred_df.to_csv(OUT_DIR / "russian_electricity_real_kan_predictions.csv", index=False)

    err_by_hour = pred_df.groupby("hour").agg(MAE=("abs_error", "mean"), RMSE=("squared_error", lambda s: float(np.sqrt(s.mean())))).reset_index()
    err_by_hour.to_csv(OUT_DIR / "russian_electricity_real_kan_error_by_hour.csv", index=False)

    err_by_region = pred_df.groupby("region").agg(
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda s: float(np.sqrt(s.mean()))),
        mean_target=("target_price_t_plus_1", "mean"),
    ).reset_index().sort_values("RMSE", ascending=False)
    err_by_region.to_csv(OUT_DIR / "russian_electricity_real_kan_error_by_region.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.barplot(data=err_by_hour, x="hour", y="MAE", color="#4d908e")
    plt.title(f"Ошибка {best_model_name} по часу суток")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_real_kan_mae_by_hour.png", dpi=160)
    plt.close()

    top_regions = err_by_region.head(12)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_regions, y="region", x="RMSE", color="#577590")
    plt.title(f"Топ регионов по RMSE для {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_real_kan_top_region_rmse.png", dpi=160)
    plt.close()

    sample_regions = list(pred_df["region"].drop_duplicates()[:3])
    vis = pred_df[pred_df["region"].isin(sample_regions)].sort_values(["region", "datetime"]).groupby("region").head(72)
    plt.figure(figsize=(12, 6))
    for region, sub in vis.groupby("region"):
        plt.plot(sub["datetime"].to_numpy(), sub["target_price_t_plus_1"].to_numpy(), label=f"{region} fact", linewidth=2)
        plt.plot(sub["datetime"].to_numpy(), sub["prediction"].to_numpy(), linestyle="--", label=f"{region} pred")
    plt.xticks(rotation=30)
    plt.title(f"Факт и прогноз ({best_model_name})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_real_kan_actual_vs_pred.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.hist(pred_df["residual"].to_numpy(), bins=60, color="#bc4749", alpha=0.85)
    plt.title(f"Распределение остатков {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_real_kan_residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=train_history, x="epoch", y="val_rmse", hue="model", marker="o")
    plt.title("Динамика валидационной ошибки KAN-моделей")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "russian_electricity_real_kan_training_curves.png", dpi=160)
    plt.close()

    if "KAN" in models:
        kan_importance = save_phi_plots(models["KAN"], ds, "KAN")
        kan_importance.to_csv(OUT_DIR / "russian_electricity_kan_feature_importance.csv", index=False)
    if "HybridKAN" in models:
        hybrid_importance = save_phi_plots(models["HybridKAN"], ds, "HybridKAN")
        hybrid_importance.to_csv(OUT_DIR / "russian_electricity_hybridkan_feature_importance.csv", index=False)
        group_imp = permutation_importance_groups(models["HybridKAN"], ds, ds["feature_groups"])
        group_imp.to_csv(OUT_DIR / "russian_electricity_hybridkan_channel_importance.csv", index=False)
        plt.figure(figsize=(9, 4))
        sns.barplot(data=group_imp, x="group", y="rmse_delta", color="#f4a261")
        plt.title("Важность каналов признаков (perm. importance, HybridKAN)")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "russian_electricity_hybridkan_channel_importance.png", dpi=160)
        plt.close()

    return pred_df, err_by_hour, err_by_region


def save_summary(meta, ds, metrics, pred_df):
    kan_family = metrics[metrics["model"].isin(["KAN", "HybridKAN"])].sort_values("RMSE").reset_index(drop=True)
    summary = {
        **meta,
        "train_rows": int(len(ds["train"])),
        "val_rows": int(len(ds["val"])),
        "test_rows": int(len(ds["test"])),
        "target_mean": float(ds["train"]["target_price_t_plus_1"].mean()),
        "target_std": float(ds["train"]["target_price_t_plus_1"].std()),
        "best_model_overall": str(metrics.iloc[0]["model"]),
        "best_rmse_overall": float(metrics.iloc[0]["RMSE"]),
        "best_kan_model": str(kan_family.iloc[0]["model"]) if not kan_family.empty else "",
        "best_kan_rmse": float(kan_family.iloc[0]["RMSE"]) if not kan_family.empty else None,
    }
    (OUT_DIR / "russian_electricity_real_kan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top_channel_path = OUT_DIR / "russian_electricity_hybridkan_channel_importance.csv"
    top_feature_path = OUT_DIR / "russian_electricity_hybridkan_feature_importance.csv"
    channel_df = pd.read_csv(top_channel_path) if top_channel_path.exists() else pd.DataFrame()
    feature_df = pd.read_csv(top_feature_path) if top_feature_path.exists() else pd.DataFrame()

    lines = [
        "## Обновлённый эксперимент по электроэнергии РФ",
        "",
        "- Постановка исправлена на leakage-safe прогнозирование цены на 1 час вперёд (`t+1`).",
        "- Суточные потери используются только с лагом 1 день, месячные тарифные показатели — только с лагом 1 месяц.",
        f"- Итоговая панель: {meta['panel_rows']} наблюдений, {meta['regions_in_panel']} региона, период {meta['datetime_min']} — {meta['datetime_max']}.",
        "",
        "## Метрики моделей",
        "",
        metrics.to_string(index=False),
        "",
    ]

    if not channel_df.empty:
        lines += [
            "## Наиболее важные каналы для HybridKAN",
            "",
            channel_df.head(5).to_string(index=False),
            "",
        ]
    if not feature_df.empty:
        lines += [
            "## Наиболее важные признаки первого KAN-слоя",
            "",
            feature_df.head(10).to_string(index=False),
            "",
        ]

    worst_region = pred_df.groupby("region")["abs_error"].mean().sort_values(ascending=False).head(1)
    if not worst_region.empty:
        region = worst_region.index[0]
        value = float(worst_region.iloc[0])
        lines += [
            "## Короткая интерпретация",
            "",
            f"- Лучшая модель overall: `{metrics.iloc[0]['model']}` с RMSE `{metrics.iloc[0]['RMSE']:.3f}`.",
            f"- Лучшая модель семейства KAN: `{kan_family.iloc[0]['model']}` с RMSE `{kan_family.iloc[0]['RMSE']:.3f}`." if not kan_family.empty else "- Модели семейства KAN не найдены в таблице метрик.",
            f"- Наиболее сложный регион по средней абсолютной ошибке: `{region}` (`{value:.2f}`).",
            "- `phi`-функции сохранены отдельно для визуального анализа нелинейностей первого KAN-слоя.",
        ]

    (OUT_DIR / "russian_electricity_real_kan_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    set_seed(SEED)
    panel, meta = load_or_build_panel()
    panel.to_csv(OUT_DIR / "russian_electricity_panel_dataset_leakage_safe.csv", index=False)

    ds = prepare_datasets(panel)
    metrics, preds, models, train_history = evaluate_models(ds)
    metrics.to_csv(OUT_DIR / "russian_electricity_real_kan_metrics.csv", index=False)
    train_history.to_csv(OUT_DIR / "russian_electricity_real_kan_train_history.csv", index=False)

    pred_df, _, _ = save_diagnostics(ds, preds, metrics, models, train_history)
    save_summary(meta, ds, metrics, pred_df)

    print(metrics.to_string(index=False))
    print("Saved leakage-safe panel to:", OUT_DIR / "russian_electricity_panel_dataset_leakage_safe.csv")
    print("Saved metrics to:", OUT_DIR / "russian_electricity_real_kan_metrics.csv")


if __name__ == "__main__":
    main()
