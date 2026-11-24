"""
Sales Volume Forecasting with a Neural Network
==============================================

Predicts next-month order quantity for ten high-volume inventory items using
sales order lines from 2019-01 to 2024-10.

Pipeline
  1. Load the order lines from Excel.
  2. Clean the data (whitespace, partial month, zero quantities,
     re-issued duplicate lines). Spikes are handled by a log transform.
  3. Aggregate to monthly quantity per item.
  4. Build supervised features (12 monthly lags, rolling means, month-of-year,
     item identity) and normalise them.
  5. Train and compare five versions of a feed-forward neural network
     (scikit-learn MLPRegressor) against simple baselines on the last
     12 months (Oct 2023 - Sep 2024), which the models never see in training.
  6. Retrain the final model on all data and forecast Oct 2024 - Mar 2025.

Usage
  python sales_forecast.py [path_to_excel]
  With no argument, it uses the Sales_orders*.xlsx file in the script's folder.

Outputs (written to an "output" folder next to this script):
  metrics_overall.csv, metrics_by_item.csv, test_predictions.csv,
  future_forecast.csv, and PNG charts.

Requires: pandas, numpy, scikit-learn, matplotlib, openpyxl
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Paths are relative to this script's folder, so it works no matter which
# folder it is run from (e.g. the Run button in an IDE).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")


def find_data_file():
    """Use the path given on the command line, otherwise look for an Excel
    file starting with 'Sales_orders' in the same folder as this script."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    matches = sorted(f for f in os.listdir(SCRIPT_DIR)
                     if f.lower().startswith("sales_orders") and f.lower().endswith(".xlsx"))
    if not matches:
        sys.exit("Could not find the sales Excel file. Put 'Sales_orders_2019-2024.xlsx' "
                 f"in this folder:\n  {SCRIPT_DIR}\n"
                 "or run:  python sales_forecast.py \"C:\\path\\to\\file.xlsx\"")
    return os.path.join(SCRIPT_DIR, matches[0])


DATA_FILE = find_data_file()

# Ten real inventory items with orders in every month of the history
ITEMS = ["10445", "71061", "70398", "70553", "71159",
         "83647", "70443", "35150", "70873", "70554"]

N_LAGS = 12                 # months of history fed to the network
TEST_MONTHS = 12            # hold-out period (Oct 2023 - Sep 2024)
FORECAST_MONTHS = 6         # future horizon after the data ends
LAST_FULL_MONTH = "2024-09"  # Oct 2024 has only 8 lines, so it is dropped
SEEDS = list(range(10))     # ensemble members for the final model

# One purchase order (28 Nov 2023) bought 969 units of six
# of the selected parts in a single stocking order - about 7x a normal month.
# It is kept in the data (it is real demand) but results are also reported
# without that month, and it is excluded from the seasonality chart.
BULK_ORDER_NO = 94556
BULK_MONTH = "2023-11"

os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_excel(path)
    print(f"Loaded {len(df):,} order lines, {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_data(df):
    log = {"rows_loaded": len(df)}
    df = df.copy()

    # Text fields carry trailing spaces; item numbers mix ints and strings
    df["Item Number"] = df["Item Number"].astype(str).str.strip()
    for c in ["Item Description 1", "Item Description 2"]:
        df[c] = df[c].astype(str).str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Drop the incomplete final month (only a handful of lines in Oct 2024)
    end = pd.Period(LAST_FULL_MONTH, "M").to_timestamp(how="end")
    n = len(df)
    df = df[df["Order Date"] <= end]
    log["partial_month_removed"] = n - len(df)

    # Lines with zero quantity carry no demand
    n = len(df)
    df = df[df["Quantity Ordered"] > 0]
    log["zero_qty_removed"] = n - len(df)

    # Re-issued lines: salesperson 1 re-enters an order (same date, customer,
    # item and quantity, adjacent order number). The original line is left
    # with Quantity to Ship = 0. Keep only one copy so demand isn't doubled.
    reissue = df[df["Salesperson Number"] == 1]
    key = ["Order Date", "Customer Number", "Item Number", "Quantity Ordered"]
    m = df.reset_index().merge(reissue[key + ["ord_no"]], on=key, suffixes=("", "_r"))
    dup_idx = m[(m["Salesperson Number"] != 1)
                & ((m["ord_no"] - m["ord_no_r"]).abs() <= 3)]["index"].unique()
    df = df.drop(index=dup_idx)
    log["reissued_duplicates_removed"] = len(dup_idx)

    log["rows_after_cleaning"] = len(df)
    return df, log


# --------------------------------------------------------------------------
# 3. Monthly series
# --------------------------------------------------------------------------
def monthly_series(df, items):
    d = df[df["Item Number"].isin(items)].copy()
    d["Month"] = d["Order Date"].dt.to_period("M")
    s = d.pivot_table(index="Month", columns="Item Number",
                      values="Quantity Ordered", aggfunc="sum", fill_value=0)
    full = pd.period_range(s.index.min(), LAST_FULL_MONTH, freq="M")
    return s.reindex(full, fill_value=0)[items].astype(float)


# --------------------------------------------------------------------------
# 4. Features
# --------------------------------------------------------------------------
class ItemScaler:
    """Per-item normalisation fitted on training months only.
    Optional log1p transform first, then z-score."""

    def __init__(self, use_log):
        self.use_log = use_log
        self.mu, self.sd = {}, {}

    def _t(self, x):
        return np.log1p(x) if self.use_log else x

    def fit(self, series_train):
        for c in series_train:
            v = self._t(series_train[c].values)
            self.mu[c], self.sd[c] = v.mean(), v.std() + 1e-9
        return self

    def transform(self, item, x):
        return (self._t(np.asarray(x, float)) - self.mu[item]) / self.sd[item]

    def inverse(self, item, z):
        v = np.asarray(z) * self.sd[item] + self.mu[item]
        return np.clip(np.expm1(v) if self.use_log else v, 0, None)


def make_features(z_hist, month, item_idx, n_items, extra):
    """z_hist: the last N_LAGS normalised values (oldest first)."""
    f = list(z_hist)
    if extra:
        f += [z_hist[-3:].mean(), z_hist[-6:].mean(), z_hist.mean(), z_hist.std()]
        f += [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
    if n_items > 1:
        one_hot = np.zeros(n_items)
        one_hot[item_idx] = 1
        f += list(one_hot)
    return f


def build_xy(series, scaler, items, start_pos, end_pos, extra, pooled):
    """Rows for target positions start_pos..end_pos-1 of the monthly index."""
    X, y, meta = [], [], []
    for i, item in enumerate(items):
        z = scaler.transform(item, series[item].values)
        for t in range(max(start_pos, N_LAGS), end_pos):
            X.append(make_features(z[t - N_LAGS:t], series.index[t].month, i,
                                   len(items) if pooled else 1, extra))
            y.append(z[t])
            meta.append((item, series.index[t]))
    return np.array(X), np.array(y), meta


# --------------------------------------------------------------------------
# 5. Models
# --------------------------------------------------------------------------
def make_mlp(hidden, alpha, seed):
    return MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                        solver="adam", alpha=alpha, learning_rate_init=1e-3,
                        max_iter=2000, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=50,
                        random_state=seed)


def train_ensemble(Xtr, ytr, cfg, seeds):
    fs = StandardScaler().fit(Xtr)
    models = [make_mlp(cfg["hidden"], cfg["alpha"], s).fit(fs.transform(Xtr), ytr)
              for s in seeds]
    return fs, models


def ensemble_predict(fs, models, X):
    return np.mean([m.predict(fs.transform(X)) for m in models], axis=0)


def bias_shift(scaler, fs, models, Xtr, ytr, meta_tr, items):
    """Log-normal bias correction. A network trained on log values predicts
    something close to the median, which under-forecasts the mean. Adding
    half the in-sample residual variance (in log units) corrects this:
    E[Y] = exp(mu + sigma^2 / 2). Returns a shift per item in z units."""
    resid = ytr - ensemble_predict(fs, models, Xtr)
    shift = {}
    for item in items:
        r = resid[[m[0] == item for m in meta_tr]] * scaler.sd[item]
        shift[item] = 0.5 * r.var() / scaler.sd[item]
    return shift


def fit_predict(series, items, n_train, cfg, seeds):
    """Train on months [0, n_train) and make one-step-ahead predictions for
    months n_train..end, always using actual past values as inputs."""
    scaler = ItemScaler(cfg["log"]).fit(series.iloc[:n_train])
    n_total = len(series)
    preds = {}
    groups = [items] if cfg["pooled"] else [[it] for it in items]
    for grp in groups:
        Xtr, ytr, meta_tr = build_xy(series, scaler, grp, 0, n_train, cfg["extra"], cfg["pooled"])
        Xte, _, meta = build_xy(series, scaler, grp, n_train, n_total, cfg["extra"], cfg["pooled"])
        fs, models = train_ensemble(Xtr, ytr, cfg, seeds)
        zp = ensemble_predict(fs, models, Xte)
        shift = (bias_shift(scaler, fs, models, Xtr, ytr, meta_tr, grp)
                 if cfg.get("bias_corr") else {i: 0.0 for i in grp})
        for (item, month), z in zip(meta, zp):
            preds[(item, month)] = float(scaler.inverse(item, z + shift[item]))
    out = pd.DataFrame(index=series.index[n_train:], columns=items, dtype=float)
    for (item, month), v in preds.items():
        out.loc[month, item] = v
    return out


def baselines(series, n_train):
    test_idx = series.index[n_train:]
    b = {}
    b["Naive (last month)"] = series.shift(1).loc[test_idx]
    b["Seasonal naive (same month last year)"] = series.shift(12).loc[test_idx]
    b["3-month moving average"] = series.rolling(3).mean().shift(1).loc[test_idx]
    b["12-month moving average"] = series.rolling(12).mean().shift(1).loc[test_idx]
    return b


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(actual, pred):
    a, p = actual.values.ravel(), pred.values.ravel()
    e = p - a
    return {"MAE": np.mean(np.abs(e)),
            "RMSE": np.sqrt(np.mean(e ** 2)),
            "WAPE %": 100 * np.abs(e).sum() / a.sum(),
            "Bias %": 100 * e.sum() / a.sum()}


def score(actual, pred):
    """Metrics on the full test year, and on the test year without the
    one-off bulk-order month."""
    out = metrics(actual, pred)
    keep = actual.index != pd.Period(BULK_MONTH, "M")
    ex = metrics(actual[keep], pred[keep])
    out["WAPE % excl. bulk month"] = ex["WAPE %"]
    out["Bias % excl. bulk month"] = ex["Bias %"]
    return out


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def save_eda_charts(df, series, descs):
    m = df.groupby(df["Order Date"].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(m.index.to_timestamp(), m.values, color="#2a6f97")
    ax.set_title("Order lines per month (all items, after cleaning)")
    ax.set_ylabel("Order lines")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/eda_orders_per_month.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(11, 12), sharex=True)
    for ax, item in zip(axes.ravel(), series.columns):
        ax.plot(series.index.to_timestamp(), series[item], color="#2a6f97", lw=1.2)
        ax.set_title(f"{item}  {descs[item]}", fontsize=9)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=8)
    fig.suptitle("Monthly quantity ordered, 10 selected items", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/eda_item_series.png", dpi=150)
    plt.close(fig)

    # seasonality without the one-off bulk order, so it does not fake a November peak
    regular = monthly_series(df[df["ord_no"] != BULK_ORDER_NO], list(series.columns))
    norm = regular / regular.mean()
    seas = norm.groupby(norm.index.month).mean().mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(range(1, 13), seas.values, color="#2a6f97")
    ax.axhline(1, color="#333", lw=.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Index (1.0 = average month)")
    ax.set_title("Average seasonal pattern across the 10 items")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/eda_seasonality.png", dpi=150)
    plt.close(fig)


def save_result_charts(series, final_pred, best_base, base_name, future, descs):
    fig, axes = plt.subplots(5, 2, figsize=(11, 12))
    hist = series.iloc[-30:]
    for ax, item in zip(axes.ravel(), series.columns):
        ax.plot(hist.index.to_timestamp(), hist[item], color="#333", lw=1.2, label="Actual")
        ax.plot(final_pred.index.to_timestamp(), final_pred[item], color="#d1495b",
                lw=1.6, marker="o", ms=3, label="Neural network")
        ax.plot(best_base.index.to_timestamp(), best_base[item], color="#8a8a8a",
                lw=1, ls="--", label=base_name)
        ax.plot(future.index.to_timestamp(), future[item], color="#d1495b", lw=1.6,
                ls=":", marker="o", ms=3, label="Future forecast")
        ax.axvspan(final_pred.index[0].to_timestamp(), final_pred.index[-1].to_timestamp(),
                   color="#f3d9dc", alpha=.35)
        ax.set_title(f"{item}  {descs[item]}", fontsize=9)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Test period (shaded) and 6-month forecast", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/results_by_item.png", dpi=150)
    plt.close(fig)


def save_version_chart(overall):
    d = overall.sort_values("WAPE % excl. bulk month")
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["WAPE %"], 0.4, color="#9aa5b1", label="Full test year")
    ax.barh(y + 0.2, d["WAPE % excl. bulk month"], 0.4, color="#d1495b",
            label="Excluding Nov-2023 bulk order")
    for i, (v1, v2) in enumerate(zip(d["WAPE %"], d["WAPE % excl. bulk month"])):
        ax.text(v1 + 1, i - 0.2, f"{v1:.0f}%", va="center", fontsize=7)
        ax.text(v2 + 1, i + 0.2, f"{v2:.0f}%", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("WAPE on the 12-month test period (lower is better)")
    ax.set_title("Model versions vs. baselines")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/results_model_comparison.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. Future forecast (recursive)
# --------------------------------------------------------------------------
def forecast_future(series, items, cfg, seeds, horizon):
    scaler = ItemScaler(cfg["log"]).fit(series)
    Xtr, ytr, meta_tr = build_xy(series, scaler, items, 0, len(series), cfg["extra"], True)
    fs, models = train_ensemble(Xtr, ytr, cfg, seeds)
    shift = (bias_shift(scaler, fs, models, Xtr, ytr, meta_tr, items)
             if cfg.get("bias_corr") else {i: 0.0 for i in items})
    months = pd.period_range(series.index[-1] + 1, periods=horizon, freq="M")
    out = pd.DataFrame(index=months, columns=items, dtype=float)
    for i, item in enumerate(items):
        z = list(scaler.transform(item, series[item].values))
        for mth in months:
            x = make_features(np.array(z[-N_LAGS:]), mth.month, i, len(items), cfg["extra"])
            zp = ensemble_predict(fs, models, [x])[0]
            out.loc[mth, item] = float(scaler.inverse(item, zp + shift[item]))
            z.append(zp)  # feed the (median) prediction back in as next month's lag
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    raw = load_data(DATA_FILE)
    df, log = clean_data(raw)
    print("Cleaning:", log)

    descs = (df[df["Item Number"].isin(ITEMS)].groupby("Item Number")
             ["Item Description 2"].first().to_dict())
    series = monthly_series(df, ITEMS)
    print(f"Monthly series: {series.index[0]} to {series.index[-1]} "
          f"({len(series)} months x {len(ITEMS)} items)")

    summary = pd.DataFrame({
        "Description": [descs[i] for i in ITEMS],
        "Total qty": series.sum().values,
        "Monthly mean": series.mean().round(1).values,
        "Monthly std": series.std().round(1).values,
        "CV": (series.std() / series.mean()).round(2).values,
        "Zero months": (series == 0).sum().values,
    }, index=ITEMS)
    summary.to_csv(f"{OUT_DIR}/item_summary.csv")
    print(summary)
    save_eda_charts(df, series, descs)

    n_train = len(series) - TEST_MONTHS
    actual = series.iloc[n_train:]

    # Model versions, in the order they were tried
    versions = {
        "NN v1: one network per item, raw scale":
            dict(pooled=False, log=False, extra=False, hidden=(32,), alpha=1e-4),
        "NN v2: one pooled network, per-item scaling":
            dict(pooled=True, log=False, extra=False, hidden=(32,), alpha=1e-4),
        "NN v3: + log transform, seasonality & rolling features":
            dict(pooled=True, log=True, extra=True, hidden=(32,), alpha=1e-4),
        "NN v4 (final): smaller network (16), 10-seed ensemble":
            dict(pooled=True, log=True, extra=True, hidden=(16,), alpha=1e-3),
        "NN v5 (rejected): + log-normal bias correction":
            dict(pooled=True, log=True, extra=True, hidden=(16,), alpha=1e-3,
                 bias_corr=True),
    }

    results, preds = {}, {}
    for name, cfg in versions.items():
        seeds = [0] if name.startswith(("NN v1", "NN v2", "NN v3")) else SEEDS
        p = fit_predict(series, ITEMS, n_train, cfg, seeds)
        preds[name] = p
        results[name] = score(actual, p)
        r = results[name]
        print(f"{name:55s} WAPE {r['WAPE %']:6.1f}%  bias {r['Bias %']:+6.1f}%  | "
              f"excl. bulk month: WAPE {r['WAPE % excl. bulk month']:6.1f}%  "
              f"bias {r['Bias % excl. bulk month']:+6.1f}%")

    for name, p in baselines(series, n_train).items():
        preds[name] = p
        results[name] = score(actual, p)
        r = results[name]
        print(f"{name:55s} WAPE {r['WAPE %']:6.1f}%  bias {r['Bias %']:+6.1f}%  | "
              f"excl. bulk month: WAPE {r['WAPE % excl. bulk month']:6.1f}%  "
              f"bias {r['Bias % excl. bulk month']:+6.1f}%")

    overall = pd.DataFrame(results).T.round(2)
    overall.to_csv(f"{OUT_DIR}/metrics_overall.csv")
    save_version_chart(overall)

    final_name = [n for n in versions if "final" in n][0]
    base_names = [n for n in overall.index if not n.startswith("NN")]
    best_base = overall.loc[base_names, "WAPE %"].idxmin()

    by_item = []
    keep = actual.index != pd.Period(BULK_MONTH, "M")
    for item in ITEMS:
        nn = metrics(actual[[item]], preds[final_name][[item]])
        bb = metrics(actual[[item]], preds[best_base][[item]])
        nn_ex = metrics(actual.loc[keep, [item]], preds[final_name].loc[keep, [item]])
        bb_ex = metrics(actual.loc[keep, [item]], preds[best_base].loc[keep, [item]])
        by_item.append({"Item": item, "Description": descs[item],
                        "Test actual total": actual[item].sum(),
                        "NN predicted total": round(preds[final_name][item].sum(), 1),
                        "NN MAE": round(nn["MAE"], 2),
                        "NN WAPE %": round(nn["WAPE %"], 1),
                        "Baseline WAPE %": round(bb["WAPE %"], 1),
                        "NN WAPE % excl. bulk": round(nn_ex["WAPE %"], 1),
                        "Baseline WAPE % excl. bulk": round(bb_ex["WAPE %"], 1)})
    by_item = pd.DataFrame(by_item)
    by_item.to_csv(f"{OUT_DIR}/metrics_by_item.csv", index=False)
    print(by_item)

    tp = pd.concat({"actual": actual, "nn_final": preds[final_name],
                    "best_baseline": preds[best_base]}, axis=1)
    tp.to_csv(f"{OUT_DIR}/test_predictions.csv")

    future = forecast_future(series, ITEMS, versions[final_name], SEEDS, FORECAST_MONTHS)
    future.round(1).to_csv(f"{OUT_DIR}/future_forecast.csv")
    print("Forecast Oct 2024 - Mar 2025:\n", future.round(1))

    save_result_charts(series, preds[final_name], preds[best_base], best_base, future, descs)
    pd.Series(log).to_csv(f"{OUT_DIR}/cleaning_log.csv")
    print(f"\nBest baseline: {best_base}. Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()