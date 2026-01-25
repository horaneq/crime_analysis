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
OUTPUT_PLOTS = "output/lstm/test2021/"
OUTPUT_PREDICTIONS = "output/lstm/predictions_lstm_2021_2024.csv"
OUTPUT_METRICS = "output/lstm/metrics_lstm_2021_2024.csv"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)

SEQ_LEN = 5
EPOCHS = 400
BATCH_SIZE = 8
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["crime"]).copy()

exo_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]
df[exo_cols] = df[exo_cols].fillna(df[exo_cols].median())

units = sorted(df["unit"].unique())

TRAIN_MAX_YEAR = 2020
TEST_MIN_YEAR = 2021
TEST_MAX_YEAR = 2024

def make_sequences_exo_to_y(df_unit, seq_len):
    """
    X: sekwencje exo z ostatnich seq_len lat
    y: crime w kolejnym roku (tzn. crime w roku i)
    years_y: rok odpowiadający y
    """
    df_unit = df_unit.sort_values("year").reset_index(drop=True)

    X_list, y_list, years_list = [], [], []
    exo = df_unit[exo_cols].values.astype(np.float32)
    y = df_unit["crime"].values.astype(np.float32)
    years = df_unit["year"].values

    for i in range(seq_len, len(df_unit)):
        X_list.append(exo[i-seq_len:i])
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true != 0, y_true, 1e-6)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

all_pred_rows = []
all_metrics = []

for unit in units:
    df_unit = df[df["unit"] == unit].copy().sort_values("year")

    df_train = df_unit[df_unit["year"] <= TRAIN_MAX_YEAR].copy()
    df_test = df_unit[(df_unit["year"] >= TEST_MIN_YEAR) & (df_unit["year"] <= TEST_MAX_YEAR)].copy()

    if len(df_train) < SEQ_LEN + 6 or len(df_test) == 0:
        continue

    df_all_until_test_end = df_unit[df_unit["year"] <= TEST_MAX_YEAR].copy()

    X_all, y_all, years_y = make_sequences_exo_to_y(df_all_until_test_end, SEQ_LEN)

    train_mask = years_y <= TRAIN_MAX_YEAR
    test_mask = (years_y >= TEST_MIN_YEAR) & (years_y <= TEST_MAX_YEAR)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    years_test = years_y[test_mask]

    if len(X_test) == 0:
        continue

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train.reshape(-1, len(exo_cols))).reshape(X_train.shape)
    X_test_s = x_scaler.transform(X_test.reshape(-1, len(exo_cols))).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    model = build_lstm((SEQ_LEN, len(exo_cols)))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=15, factor=0.5, min_lr=1e-5)

    model.fit(
        X_train_s, y_train_s,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es, rlrop]
    )

    y_pred_s = model.predict(X_test_s, verbose=0).reshape(-1)
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(-1)
    y_pred = np.maximum(y_pred, 0)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = safe_mape(y_test, y_pred)

    all_metrics.append({
        "unit": unit,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Model": "LSTM_exo_seq"
    })

    for yr, yt, yp in zip(years_test, y_test, y_pred):
        all_pred_rows.append({
            "year": int(yr),
            "unit": unit,
            "crime_actual": float(yt),
            "crime_predicted": float(yp),
            "R2_unit": float(r2),
            "MAE_unit": float(mae),
            "RMSE_unit": float(rmse),
            "MAPE_unit": float(mape),
            "Model": "LSTM_exo_seq"
        })

    plt.figure(figsize=(10, 5))
    plt.plot(df_train["year"], df_train["crime"], marker="o", label="Dane historyczne (do 2020)")
    plt.plot(df_test["year"], df_test["crime"], marker="o", label="Rzeczywiste dane 2021-2024")
    plt.plot(years_test, y_pred, marker="s", linestyle="--", label="Predykcja LSTM 2021-2024")

    plt.title(f"Predykcja przestępczości (LSTM) - {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości / 1000 mieszkańców")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_LSTM_test_2021_2024.png", dpi=200)
    plt.close()

if len(all_pred_rows) > 0:
    pred_df = pd.DataFrame(all_pred_rows).sort_values(["unit", "year"])
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)

metrics_df = pd.DataFrame(all_metrics).sort_values("RMSE")
metrics_df.to_csv(OUTPUT_METRICS, index=False)

print("Gotowe!")
print(f"- Wykresy: {OUTPUT_PLOTS}")
print(f"- Predykcje: {OUTPUT_PREDICTIONS}")
print(f"- Metryki: {OUTPUT_METRICS}")
