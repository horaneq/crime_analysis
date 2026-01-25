import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from catboost import CatBoostRegressor

DATA_PATH = "data/processed/panel.csv"

OUT_DIR = "output/catboost_global/forecast2021_2024/plots/"
OUT_PRED = "output/catboost_global/predictions_catboost_2021_2024.csv"
OUT_METR = "output/catboost_global/metrics_catboost_2021_2024.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)

TRAIN_END = 2020
TEST_START = 2021
TEST_END = 2024

SEED = 42

exo_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

USE_EXO = True
EXO_LAG = 1

LAGS = [1, 2, 3, 4, 5]
ROLL_WINDOWS = [3, 5]

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) > 1e-6, y_true, 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["crime"]).copy()
df = df.sort_values(["unit", "year"]).reset_index(drop=True)

if USE_EXO:
    df[exo_cols] = df[exo_cols].fillna(df[exo_cols].median())

for lag in LAGS:
    df[f"crime_lag{lag}"] = df.groupby("unit")["crime"].shift(lag)

df["crime_shift1"] = df.groupby("unit")["crime"].shift(1)
for w in ROLL_WINDOWS:
    df[f"crime_roll_mean_{w}"] = (
        df.groupby("unit")["crime_shift1"]
          .rolling(window=w, min_periods=w)
          .mean()
          .reset_index(level=0, drop=True)
    )

if USE_EXO:
    for col in exo_cols:
        df[f"{col}_lag{EXO_LAG}"] = df.groupby("unit")[col].shift(EXO_LAG)

drop_cols = ["crime_shift1"]
df = df.drop(columns=drop_cols)

need_cols = [f"crime_lag{lag}" for lag in LAGS] + [f"crime_roll_mean_{w}" for w in ROLL_WINDOWS]
if USE_EXO:
    need_cols += [f"{c}_lag{EXO_LAG}" for c in exo_cols]

df_feat = df.dropna(subset=need_cols).copy()


train_df = df_feat[df_feat["year"] <= TRAIN_END].copy()
test_df = df_feat[(df_feat["year"] >= TEST_START) & (df_feat["year"] <= TEST_END)].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise RuntimeError("Brak danych train/test po feature engineering. Sprawdź zakres lat i braki w panel.csv.")

feature_cols = ["year"] + [f"crime_lag{lag}" for lag in LAGS] + [f"crime_roll_mean_{w}" for w in ROLL_WINDOWS]
if USE_EXO:
    feature_cols += [f"{c}_lag{EXO_LAG}" for c in exo_cols]
feature_cols += ["unit"]  

X_train = train_df[feature_cols]
y_train = train_df["crime"].astype(float)

X_test = test_df[feature_cols]
y_test = test_df["crime"].astype(float)

cat_features = ["unit"]

model = CatBoostRegressor(
    loss_function="RMSE",
    depth=6,
    learning_rate=0.05,
    iterations=5000,
    random_seed=SEED,
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=200, 
    l2_leaf_reg=6.0,
    subsample=0.9,
    colsample_bylevel=0.9,
    verbose=200
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test), 
    use_best_model=True
)


y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0.0)

pred_df = pd.DataFrame({
    "unit": test_df["unit"].values,
    "year": test_df["year"].astype(int).values,
    "actual_crime": y_test.values.astype(float),
    "predicted_crime": y_pred.astype(float),
    "Model": f"CatBoost_global_{'exoLag1' if USE_EXO else 'noExo'}"
}).sort_values(["unit", "year"])

pred_df.to_csv(OUT_PRED, index=False)

units = sorted(df_feat["unit"].unique())
metrics = []

for unit in units:
    pu = pred_df[pred_df["unit"] == unit].copy()
    if len(pu) == 0:
        continue

    yt = pu["actual_crime"].values
    yp = pu["predicted_crime"].values

    metrics.append({
        "unit": unit,
        "R2": r2_score(yt, yp) if len(yt) >= 2 else np.nan,
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": safe_mape(yt, yp),
        "Model": pred_df["Model"].iloc[0],
        "train_end": TRAIN_END,
        "test_years": f"{TEST_START}-{TEST_END}"
    })

    all_u = df[df["unit"] == unit].copy().sort_values("year")
    hist = all_u[all_u["year"] <= TRAIN_END]

    plt.figure(figsize=(10, 5))
    plt.plot(hist["year"], hist["crime"], marker="o", label=f"Dane historyczne (do {TRAIN_END})")
    plt.plot(pu["year"], yt, marker="o", label=f"Rzeczywiste dane {TEST_START}-{TEST_END}")
    plt.plot(pu["year"], yp, marker="s", linestyle="--", label="Predykcja CatBoost")
    plt.title(f"Predykcja przestępczości (CatBoost global) - {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości / 1000 mieszkańców")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{unit}_CATBOOST_2021_2024.png"), dpi=200)
    plt.close()

metr_df = pd.DataFrame(metrics).sort_values("unit")
metr_df.to_csv(OUT_METR, index=False)

r2_g = r2_score(pred_df["actual_crime"], pred_df["predicted_crime"]) if len(pred_df) >= 2 else np.nan
mae_g = mean_absolute_error(pred_df["actual_crime"], pred_df["predicted_crime"])
rmse_g = rmse(pred_df["actual_crime"], pred_df["predicted_crime"])
mape_g = safe_mape(pred_df["actual_crime"], pred_df["predicted_crime"])

print("Gotowe!")
print(f"- wykresy: {OUT_DIR}")
print(f"- predykcje: {OUT_PRED}")
print(f"- metryki: {OUT_METR}")
print(f"Global: R2={r2_g:.3f}, MAE={mae_g:.3f}, RMSE={rmse_g:.3f}, MAPE={mape_g:.2f}%")
print("Tip: jeśli wygląda gorzej, ustaw USE_EXO=False i porównaj.")