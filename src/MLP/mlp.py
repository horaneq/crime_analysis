import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks

DATA_PATH = "data/processed/panel.csv"

OUT_DIR = "output/mlp_global_legit_2021_2024/plots/"
OUT_PRED = "output/mlp_global_legit_2021_2024/predictions.csv"
OUT_METR = "output/mlp_global_legit_2021_2024/metrics.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)

SEED = 42
EPOCHS = 600
BATCH_SIZE = 32

TRAIN_END = 2020
TEST_YEARS = [2021, 2022, 2023, 2024]

tf.random.set_seed(SEED)
np.random.seed(SEED)

exo_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) > 1e-6, y_true, 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def build_model(n_features, n_units):
    x_in = layers.Input(shape=(n_features,), name="X")
    u_in = layers.Input(shape=(1,), dtype="int32", name="unit_id")

    emb_dim = max(4, int(np.ceil(np.sqrt(n_units))))
    u_emb = layers.Embedding(input_dim=n_units, output_dim=emb_dim, name="unit_emb")(u_in)
    u_flat = layers.Flatten()(u_emb)

    h = layers.Concatenate()([x_in, u_flat])
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Dense(32, activation="relu")(h)

    y_out = layers.Dense(1, name="delta_out")(h)

    m = models.Model(inputs=[x_in, u_in], outputs=y_out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["crime"]).copy()
df[exo_cols] = df[exo_cols].fillna(df[exo_cols].median())
df = df.sort_values(["unit", "year"]).reset_index(drop=True)

units = sorted(df["unit"].unique())
unit2id = {u: i for i, u in enumerate(units)}
id2unit = {i: u for u, i in unit2id.items()}
n_units = len(units)

LAGS_CRIME = [1, 2]
LAGS_EXO = [1, 2, 3]

X_train_list, y_train_list, u_train_list = [], [], []

for unit in units:
    d = df[df["unit"] == unit].copy().sort_values("year").reset_index(drop=True)

    max_lag = max(max(LAGS_CRIME), max(LAGS_EXO))
    if len(d) < max_lag + 5:
        continue

    for idx in range(max_lag, len(d)):
        year_t = int(d.loc[idx, "year"])
        if year_t > TRAIN_END:
            break

        feats = []

        for lag in LAGS_CRIME:
            feats.append(float(d.loc[idx - lag, "crime"]))

        for lag in LAGS_EXO:
            row = d.loc[idx - lag, exo_cols].astype(float).values.tolist()
            feats.extend(row)

        crime_t = float(d.loc[idx, "crime"])
        crime_prev = float(d.loc[idx - 1, "crime"])
        delta = crime_t - crime_prev

        X_train_list.append(feats)
        y_train_list.append(delta)
        u_train_list.append([unit2id[unit]])

X_train = np.asarray(X_train_list, dtype=np.float32)
y_train = np.asarray(y_train_list, dtype=np.float32)
u_train = np.asarray(u_train_list, dtype=np.int32)

if len(X_train) == 0:
    raise RuntimeError("Brak danych treningowych. Sprawdź zakres lat i panel.csv.")

n_features = X_train.shape[1]

x_scaler = StandardScaler()
X_train_s = x_scaler.fit_transform(X_train)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

model = build_model(n_features, n_units)

es = callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=20, factor=0.5, min_lr=1e-5)

model.fit(
    {"X": X_train_s, "unit_id": u_train},
    y_train_s,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[es, rlrop]
)

all_preds = []

for unit in units:
    d = df[df["unit"] == unit].copy().sort_values("year").reset_index(drop=True)
    if len(d) < 10:
        continue

    crime_series = d["crime"].astype(float).values.copy()
    years = d["year"].astype(int).values

    year2idx = {int(y): i for i, y in enumerate(years)}

    for t_year in TEST_YEARS:
        if t_year not in year2idx:
            continue

        idx = year2idx[t_year]
        max_lag = max(max(LAGS_CRIME), max(LAGS_EXO))
        if idx < max_lag:
            continue

        feats = []

        for lag in LAGS_CRIME:
            feats.append(float(crime_series[idx - lag]))

        for lag in LAGS_EXO:
            feats.extend(d.loc[idx - lag, exo_cols].astype(float).values.tolist())

        X_t = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        X_t_s = x_scaler.transform(X_t)

        delta_s = model.predict({"X": X_t_s, "unit_id": np.asarray([[unit2id[unit]]], dtype=np.int32)}, verbose=0).reshape(-1)[0]
        delta = float(y_scaler.inverse_transform([[delta_s]])[0, 0])

        pred = float(crime_series[idx - 1] + delta)
        pred = max(pred, 0.0)

        actual = float(d.loc[idx, "crime"])

        all_preds.append({
            "unit": unit,
            "year": int(t_year),
            "actual_crime": actual,
            "predicted_crime": pred,
            "Model": "MLP_global_delta_walkforward"
        })

        crime_series[idx] = pred 
        
pred_df = pd.DataFrame(all_preds).sort_values(["unit", "year"])
pred_df.to_csv(OUT_PRED, index=False)

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
        "Model": "MLP_global_delta_walkforward"
    })

    d = df[df["unit"] == unit].copy().sort_values("year")
    hist = d[d["year"] <= TRAIN_END]

    plt.figure(figsize=(10, 5))
    plt.plot(hist["year"], hist["crime"], marker="o", label=f"Dane historyczne (do {TRAIN_END})")
    plt.plot(pu["year"], yt, marker="o", label="Rzeczywiste dane 2021-2024")
    plt.plot(pu["year"], yp, marker="s", linestyle="--", label="Predykcja MLP 2021-2024 (walk-forward)")
    plt.title(f"Predykcja przestępczości (MLP global, delta) - {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości / 1000 mieszkańców")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{unit}_MLP_DELTA_2021_2024.png"), dpi=200)
    plt.close()

metr_df = pd.DataFrame(metrics).sort_values("unit")
metr_df.to_csv(OUT_METR, index=False)

if len(pred_df) >= 2:
    r2_g = r2_score(pred_df["actual_crime"], pred_df["predicted_crime"])
else:
    r2_g = np.nan

mae_g = mean_absolute_error(pred_df["actual_crime"], pred_df["predicted_crime"])
rmse_g = rmse(pred_df["actual_crime"], pred_df["predicted_crime"])
mape_g = safe_mape(pred_df["actual_crime"], pred_df["predicted_crime"])

print("Gotowe!")
print(f"- wykresy: {OUT_DIR}")
print(f"- predykcje: {OUT_PRED}")
print(f"- metryki: {OUT_METR}")
print(f"Global: R2={r2_g:.3f}, MAE={mae_g:.3f}, RMSE={rmse_g:.3f}, MAPE={mape_g:.2f}%")
