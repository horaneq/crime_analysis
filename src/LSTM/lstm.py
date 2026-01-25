import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
layers = tf.keras.layers
callbacks = tf.keras.callbacks
models = tf.keras.models

DATA_PATH = "data/processed/panel.csv"

OUT_DIR = "output/lstm/forecast2024/"
OUT_PRED = "output/lstm/predictions_lstm_2021_2024.csv"
OUT_METR = "output/lstm/metrics_lstm_2021_2024.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)

SEQ_LEN = 5
EPOCHS = 400
BATCH_SIZE = 8
SEED = 42

TRAIN_END = 2020
TEST_START = 2021
TEST_END = 2024

tf.random.set_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["crime"]).copy()

exo_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

df[exo_cols] = df[exo_cols].fillna(df[exo_cols].median())

units = sorted(df["unit"].unique())

def make_sequences_exo_to_y(df_unit: pd.DataFrame, seq_len: int):
    """
    X: sekwencje exo z ostatnich seq_len lat
    y: crime w roku i
    years: rok odpowiadający y
    """
    df_unit = df_unit.sort_values("year").reset_index(drop=True)
    X_list, y_list, years_list = [], [], []

    exo = df_unit[exo_cols].values.astype(np.float32)
    y = df_unit["crime"].values.astype(np.float32)
    years = df_unit["year"].values.astype(int)

    for i in range(seq_len, len(df_unit)):
        X_list.append(exo[i - seq_len:i])
        y_list.append(y[i])
        years_list.append(years[i])

    return np.array(X_list), np.array(y_list), np.array(years_list)

def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return model

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) > 1e-6, y_true, 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

all_pred_rows = []
all_metrics = []

for unit in units:
    df_unit = df[df["unit"] == unit].copy().sort_values("year")

    df_train = df_unit[df_unit["year"] <= TRAIN_END].copy()
    df_test = df_unit[(df_unit["year"] >= TEST_START) & (df_unit["year"] <= TEST_END)].copy()

    if len(df_train) < SEQ_LEN + 6:
        continue
    if len(df_test) == 0:
        continue

    X_tr, y_tr, _ = make_sequences_exo_to_y(df_train, SEQ_LEN)

    x_scaler = StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr.reshape(-1, len(exo_cols))).reshape(X_tr.shape)

    y_scaler = StandardScaler()
    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).reshape(-1)

    model = build_lstm((SEQ_LEN, len(exo_cols)))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=15, factor=0.5, min_lr=1e-5)

    model.fit(
        X_tr_s, y_tr_s,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es, rlrop]
    )

    exo_all = df_unit[["year"] + exo_cols].copy().sort_values("year").reset_index(drop=True)

    y_pred_list = []
    years_pred = []

    for yr in sorted(df_test["year"].unique()):
        window = exo_all[exo_all["year"] < yr].tail(SEQ_LEN)
        if len(window) < SEQ_LEN:
            continue

        X_in = window[exo_cols].values.astype(np.float32)
        X_in_s = x_scaler.transform(X_in).reshape(1, SEQ_LEN, len(exo_cols))

        y_hat_s = float(model.predict(X_in_s, verbose=0).reshape(-1)[0])
        y_hat = float(y_scaler.inverse_transform([[y_hat_s]])[0, 0])
        y_hat = max(y_hat, 0.0)

        years_pred.append(int(yr))
        y_pred_list.append(y_hat)

    if len(years_pred) == 0:
        continue

    df_test_used = df_test[df_test["year"].isin(years_pred)].copy().sort_values("year")
    y_true = df_test_used["crime"].values.astype(float)
    y_pred = np.array(y_pred_list, dtype=float)

    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = safe_mape(y_true, y_pred)

    all_metrics.append({
        "unit": unit,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Model": "LSTM_exo_only",
        "train_end": TRAIN_END,
        "test_years": f"{TEST_START}-{TEST_END}"
    })

    for yr, yt, yp in zip(df_test_used["year"].values.astype(int), y_true, y_pred):
        all_pred_rows.append({
            "unit": unit,
            "year": int(yr),
            "actual_crime": float(yt),
            "predicted_crime": float(yp),
            "R2_unit": r2,
            "MAE_unit": mae,
            "RMSE_unit": rmse,
            "MAPE_unit": mape,
            "Model": "LSTM_exo_only"
        })

    plt.figure(figsize=(10, 5))
    plt.plot(df_train["year"], df_train["crime"], marker="o", label=f"Dane historyczne (do {TRAIN_END})")
    plt.plot(df_test_used["year"], y_true, marker="o", label=f"Rzeczywiste dane {TEST_START}-{TEST_END}")
    plt.plot(df_test_used["year"], y_pred, marker="s", linestyle="--", label=f"Predykcja LSTM {TEST_START}-{TEST_END}")
    plt.title(f"Predykcja przestępczości (LSTM) - {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości / 1000 mieszkańców")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{unit}_LSTM_2021_2024.png"), dpi=200)
    plt.close()

if len(all_pred_rows) == 0:
    print("Brak wyników - sprawdź czy masz dane 2002-2024 dla województw i czy po filtrowaniu nie ubyło lat.")
else:
    pred_df = pd.DataFrame(all_pred_rows).sort_values(["unit", "year"])
    pred_df.to_csv(OUT_PRED, index=False)

    metr_df = pd.DataFrame(all_metrics).sort_values("unit")
    metr_df.to_csv(OUT_METR, index=False)

    print("Gotowe!")
    print(f"- wykresy: {OUT_DIR}")
    print(f"- predykcje: {OUT_PRED}")
    print(f"- metryki: {OUT_METR}")
