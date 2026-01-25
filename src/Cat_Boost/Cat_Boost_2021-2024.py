import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool

DATA_PATH = "data/processed/panel.csv"

OUT_DIR = "output/catboost_delta_2021_2024/plots/"
OUT_PRED = "output/catboost_delta_2021_2024/predictions.csv"
OUT_METR = "output/catboost_delta_2021_2024/metrics.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)

TRAIN_END = 2020
TEST_START = 2021
TEST_END = 2024

N_LAGS = 5
SEED = 42
np.random.seed(SEED)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_lags(df_unit: pd.DataFrame, n_lags: int):
    df_u = df_unit.sort_values("year").copy()
    for k in range(1, n_lags + 1):
        df_u[f"crime_lag{k}"] = df_u["crime"].shift(k)
    return df_u

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["crime"]).copy()

units = sorted(df["unit"].unique())

dfs = []
for u in units:
    d = df[df["unit"] == u].copy()
    d = make_lags(d, N_LAGS)
    dfs.append(d)

df_lag = pd.concat(dfs, ignore_index=True)

lag_cols = [f"crime_lag{k}" for k in range(1, N_LAGS + 1)]
df_lag = df_lag.dropna(subset=lag_cols).copy()

df_lag["delta"] = df_lag["crime"] - df_lag["crime_lag1"]

num_cols = ["year"] + lag_cols
cat_col = "unit"

train_df = df_lag[df_lag["year"] <= TRAIN_END].copy()
test_df = df_lag[(df_lag["year"] >= TEST_START) & (df_lag["year"] <= TEST_END)].copy()

X_train = train_df[num_cols + [cat_col]].copy()
y_train = train_df["delta"].values.astype(float)

X_train[cat_col] = X_train[cat_col].astype(str)

cat_features = [X_train.columns.get_loc(cat_col)]

train_pool = Pool(X_train, y_train, cat_features=cat_features)

model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=1500,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=10,
    random_seed=SEED,
    verbose=200,
    early_stopping_rounds=200
)

model.fit(train_pool)

best_it = model.get_best_iteration()
print(f"Best iteration: {best_it}")


preds = []

for unit in units:
    hist = df[(df["unit"] == unit) & (df["year"] <= TRAIN_END)].sort_values("year").copy()
    hist = hist[["year", "crime"]].reset_index(drop=True)

    for yr in range(TEST_START, TEST_END + 1):
        if len(hist) < N_LAGS:
            continue

        last_crime = float(hist["crime"].iloc[-1])

        row = {"year": int(yr)}
        for k in range(1, N_LAGS + 1):
            row[f"crime_lag{k}"] = float(hist["crime"].iloc[-k])

        row[cat_col] = str(unit)

        X_one = pd.DataFrame([row], columns=num_cols + [cat_col])
        one_pool = Pool(X_one, cat_features=cat_features)

        delta_hat = float(model.predict(one_pool)[0])
        crime_hat = max(last_crime + delta_hat, 0.0)

        actual_vals = df[(df["unit"] == unit) & (df["year"] == yr)]["crime"].values
        if len(actual_vals) == 0:
            actual = np.nan
        else:
            actual = float(actual_vals[0])

        preds.append({
            "unit": unit,
            "year": int(yr),
            "actual": actual,
            "predicted": float(crime_hat)
        })

        hist = pd.concat([hist, pd.DataFrame([{"year": int(yr), "crime": float(crime_hat)}])], ignore_index=True)

pred_df = pd.DataFrame(preds).sort_values(["unit", "year"])
pred_df.to_csv(OUT_PRED, index=False)

metrics = []

for unit in units:
    d = pred_df[pred_df["unit"] == unit].dropna(subset=["actual"]).copy()
    if len(d) == 0:
        continue

    r2 = r2_score(d["actual"], d["predicted"]) if len(d) >= 2 else np.nan
    mae = mean_absolute_error(d["actual"], d["predicted"])
    _rmse = rmse(d["actual"], d["predicted"])

    metrics.append({"unit": unit, "R2": r2, "MAE": mae, "RMSE": _rmse})

    hist = df[(df["unit"] == unit) & (df["year"] <= TRAIN_END)].sort_values("year")

    plt.figure(figsize=(10, 5))
    plt.plot(hist["year"], hist["crime"], label=f"Historia do {TRAIN_END}", marker="o")
    plt.plot(d["year"], d["actual"], label=f"Rzeczywiste {TEST_START}-{TEST_END}", marker="o")
    plt.plot(d["year"], d["predicted"], label="Predykcja", marker="s", linestyle="--")
    plt.title(f"CatBoost (delta) – {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Przestępczość / 1000")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{unit}.png"), dpi=200)
    plt.close()

metr_df = pd.DataFrame(metrics).sort_values("unit")
metr_df.to_csv(OUT_METR, index=False)

pred_global = pred_df.dropna(subset=["actual"]).copy()
if len(pred_global) >= 2:
    r2_g = r2_score(pred_global["actual"], pred_global["predicted"])
else:
    r2_g = np.nan
mae_g = mean_absolute_error(pred_global["actual"], pred_global["predicted"])
rmse_g = rmse(pred_global["actual"], pred_global["predicted"])

print("GOTOWE.")
print(f"- wykresy: {OUT_DIR}")
print(f"- predykcje: {OUT_PRED}")
print(f"- metryki: {OUT_METR}")
print(f"Global: R2={r2_g:.3f}, MAE={mae_g:.3f}, RMSE={rmse_g:.3f}")
