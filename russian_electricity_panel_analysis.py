import os

# Reduce OpenMP-related issues on the local environment before importing numpy/sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_USE_SHM", "0")

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/Users/kirill/Desktop/ВКР ИТМО")
DATA_DIR = ROOT / "Данные электроэнергии"
OUT_DIR = ROOT / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid")
SEED = 42
np.random.seed(SEED)


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


def build_panel_dataset():
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
        "Объем потерь, МВтЧ.": "daily_losses_mwh",
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

        agg_spec = {col: "first" for col in hourly_keep}
        agg_spec["Субъект РФ - Получатель"] = "nunique"
        agg_spec["Объём перетока"] = ["sum", "mean", "max"]
        g = h.groupby(["Субъект РФ", "Дата", "Час"]).agg(agg_spec)
        g.columns = ["_".join([x for x in col if x]).strip("_") for col in g.columns]
        g = g.reset_index().rename(columns=base_rename)

        d = pd.read_csv(daily[region])
        d["Дата"] = pd.to_datetime(d["Дата"])
        d = d[["Дата", "Объем потерь, МВтЧ."]].rename(columns=base_rename)
        daily_min.append(d["Дата"].min())
        daily_max.append(d["Дата"].max())

        m = pd.read_csv(monthly[region])
        for src, dst in monthly_cols.items():
            m[dst] = m[src].map(to_num)
        m["month"] = pd.to_datetime(m["Дата_x"], dayfirst=True, errors="coerce").dt.to_period("M").dt.to_timestamp()
        month_min.append(m["month"].min())
        month_max.append(m["month"].max())
        m_agg = (
            m.groupby("month")
            .agg(
                {
                    "so_service_cost": "sum",
                    "cfr_service_cost": "sum",
                    "peak_demand_mw": "max",
                    "energy_power_cost": "max",
                    "grid_losses_volume": "max",
                    "grid_loss_tariff_rate": "max",
                }
            )
            .reset_index()
        )

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

    for lag in [1, 24, 168]:
        panel[f"price_lag_{lag}"] = panel.groupby("region")["purchase_price_rub_mwh"].shift(lag)
        panel[f"cons_lag_{lag}"] = panel.groupby("region")["planned_consumption_mwh"].shift(lag)

    panel["hour"] = panel["datetime"].dt.hour
    panel["dayofweek"] = panel["datetime"].dt.dayofweek
    panel["month_num"] = panel["datetime"].dt.month
    panel["is_weekend"] = (panel["dayofweek"] >= 5).astype(int)
    panel["sin_hour"] = np.sin(2 * np.pi * panel["hour"] / 24)
    panel["cos_hour"] = np.cos(2 * np.pi * panel["hour"] / 24)
    panel["sin_dow"] = np.sin(2 * np.pi * panel["dayofweek"] / 7)
    panel["cos_dow"] = np.cos(2 * np.pi * panel["dayofweek"] / 7)

    panel = panel.dropna(subset=["purchase_price_rub_mwh", "price_lag_168", "cons_lag_168"]).copy()
    panel = panel.drop(columns=[col for col in panel.columns if panel[col].isna().mean() == 1.0])

    meta = {
        "regions_total": len(regions),
        "common_window_start": str(common_start.date()),
        "common_window_end": str(common_end.date()),
        "panel_rows": int(len(panel)),
        "panel_regions": int(panel["region"].nunique()),
        "panel_columns": int(panel.shape[1]),
    }
    return panel, meta


def save_eda(panel: pd.DataFrame, meta: dict):
    eda = {
        **meta,
        "datetime_min": str(panel["datetime"].min()),
        "datetime_max": str(panel["datetime"].max()),
        "target_mean": float(panel["purchase_price_rub_mwh"].mean()),
        "target_std": float(panel["purchase_price_rub_mwh"].std()),
        "target_min": float(panel["purchase_price_rub_mwh"].min()),
        "target_max": float(panel["purchase_price_rub_mwh"].max()),
        "rows_per_region_median": int(panel.groupby("region").size().median()),
        "missing_pct": (panel.isna().mean().sort_values(ascending=False).head(20) * 100).round(2).to_dict(),
    }
    (OUT_DIR / "russian_electricity_eda_summary.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 4))
    panel.groupby(panel["datetime"].dt.date)["purchase_price_rub_mwh"].mean().plot()
    plt.title("Средняя цена покупки электроэнергии по дням")
    plt.xlabel("Дата")
    plt.ylabel("руб./МВт.ч.")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "price_daily_mean.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(x=panel.groupby("hour")["purchase_price_rub_mwh"].mean().index, y=panel.groupby("hour")["purchase_price_rub_mwh"].mean().values, color="#2a6f97")
    plt.title("Средняя цена по часу суток")
    plt.xlabel("Час")
    plt.ylabel("руб./МВт.ч.")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "price_by_hour.png", dpi=160)
    plt.close()

    top_regions = (
        panel.groupby("region")["purchase_price_rub_mwh"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_regions, y="region", x="purchase_price_rub_mwh", color="#468faf")
    plt.title("Топ-15 регионов по средней цене покупки")
    plt.xlabel("руб./МВт.ч.")
    plt.ylabel("Регион")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_regions_mean_price.png", dpi=160)
    plt.close()

    top_missing = (panel.isna().mean().sort_values(ascending=False).head(20) * 100).reset_index()
    top_missing.columns = ["feature", "missing_pct"]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_missing, y="feature", x="missing_pct", color="#90be6d")
    plt.title("Топ признаков по доле пропусков")
    plt.xlabel("% пропусков")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "missing_top20.png", dpi=160)
    plt.close()


class AdditiveKANRegressor:
    def __init__(self, num_basis: int = 8, alpha: float = 1.0, include_linear: bool = False):
        self.num_basis = num_basis
        self.alpha = alpha
        self.include_linear = include_linear
        self.knots_ = np.linspace(-3.0, 3.0, num_basis)
        self.model_ = Ridge(alpha=alpha)

    def _basis(self, X_num: np.ndarray) -> np.ndarray:
        basis = np.exp(-((X_num[:, :, None] - self.knots_[None, None, :]) ** 2))
        return basis.reshape(X_num.shape[0], -1)

    def _design(self, X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        parts = [self._basis(X_num)]
        if self.include_linear:
            parts.append(X_num)
        if X_cat.size:
            parts.append(X_cat)
        return np.hstack(parts)

    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.model_.fit(self._design(X_num, X_cat), y)
        return self

    def predict(self, X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._design(X_num, X_cat))


def run_models(panel: pd.DataFrame):
    train = panel[panel["datetime"] < "2023-11-01"].copy()
    val = panel[(panel["datetime"] >= "2023-11-01") & (panel["datetime"] < "2023-12-01")].copy()
    test = panel[panel["datetime"] >= "2023-12-01"].copy()

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
    dense_generation_cols = [
        "ТЭС_first",
        "ТЭС.1_first",
        "ТЭС.2_first",
        "ТЭС.3_first",
    ]
    for col in sparse_generation_cols + dense_generation_cols:
        if col in panel.columns:
            train[col] = train[col].fillna(0.0)
            val[col] = val[col].fillna(0.0)
            test[col] = test[col].fillna(0.0)

    features = [
        "region",
        "Час",
        "planned_consumption_mwh",
        "full_planned_consumption_mwh",
        "recipient_count",
        "total_outflow",
        "daily_losses_mwh",
        "so_service_cost",
        "cfr_service_cost",
        "peak_demand_mw",
        "price_lag_1",
        "price_lag_24",
        "price_lag_168",
        "cons_lag_1",
        "cons_lag_24",
        "cons_lag_168",
        "hour",
        "dayofweek",
        "month_num",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "ГЭС_first",
        "АЭС_first",
        "ТЭС_first",
        "СЭС_first",
        "ВЭС_first",
        "ТЭС.1_first",
        "ТЭС.2_first",
        "ТЭС.3_first",
    ]
    features = [col for col in features if col in panel.columns]
    cat_cols = ["region"]
    num_cols = [col for col in features if col not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = OneHotEncoder(handle_unknown="ignore")

    X_train_num = num_pipe.fit_transform(train[num_cols])
    X_val_num = num_pipe.transform(val[num_cols])
    X_test_num = num_pipe.transform(test[num_cols])

    X_train_cat = cat_pipe.fit_transform(train[cat_cols]).toarray()
    X_val_cat = cat_pipe.transform(val[cat_cols]).toarray()
    X_test_cat = cat_pipe.transform(test[cat_cols]).toarray()

    X_train = np.hstack([X_train_num, X_train_cat])
    X_val = np.hstack([X_val_num, X_val_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    y_train = train["purchase_price_rub_mwh"].values.astype(float)
    y_val = val["purchase_price_rub_mwh"].values.astype(float)
    y_test = test["purchase_price_rub_mwh"].values.astype(float)

    metrics_path = OUT_DIR / "russian_electricity_panel_metrics.csv"

    def metric_row(name: str, pred):
        return {
            "model": name,
            "RMSE": float(mean_squared_error(y_test, pred, squared=False)),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        }

    results = []

    def append_result(name: str, pred):
        results.append(metric_row(name, pred))
        pd.DataFrame(results).sort_values("RMSE").to_csv(metrics_path, index=False)

    append_result("LastValueLag1", test["price_lag_1"].values)

    stacked_X = np.vstack([X_train, X_val])
    stacked_y = np.hstack([y_train, y_val])

    linear = LinearRegression().fit(stacked_X, stacked_y)
    append_result("LinearRegression", linear.predict(X_test))

    stacked_num = np.vstack([X_train_num, X_val_num])
    stacked_cat = np.vstack([X_train_cat, X_val_cat])

    kan_model = AdditiveKANRegressor(num_basis=8, alpha=2.0, include_linear=False).fit(stacked_num, stacked_cat, stacked_y)
    hybrid_model = AdditiveKANRegressor(num_basis=8, alpha=2.0, include_linear=True).fit(stacked_num, stacked_cat, stacked_y)
    append_result("KANApprox", kan_model.predict(X_test_num, X_test_cat))
    append_result("HybridKANApprox", hybrid_model.predict(X_test_num, X_test_cat))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(metrics_path, index=False)
    return results_df


def main():
    panel, meta = build_panel_dataset()
    panel.to_csv(OUT_DIR / "russian_electricity_panel_dataset.csv", index=False)
    save_eda(panel, meta)
    results_df = run_models(panel)

    print("EDA summary saved to:", OUT_DIR / "russian_electricity_eda_summary.json")
    print("Metrics saved to:", OUT_DIR / "russian_electricity_panel_metrics.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
