import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf

DATA_PATH = "data/processed/panel.csv"
OUTPUT_PLOTS = "output/lstm/forecast2024/"
OUTPUT_PREDICTIONS = "output/lstm/forecast_lstm_2024.csv"
OUTPUT_METRICS = "output/lstm/metrics_lstm_2024.csv"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)

SEQ_LEN = 5
EPOCHS = 250
BATCH_SIZE = 8
RANDOM_SEED = 42

TRAIN_END = 2020
TEST_START = 2021
TEST_END = 2024

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

def build_lstm(seq_len: int, n_features: int) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(32, kernel_regularizer=reg, recurrent_regularizer=reg),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.Huber(delta=1.0)
    )
    return model


def make_train_sequences_exo_to_y(df_train: pd.DataFrame, seq_len: int):
    """
    X: exo okno [t-seq_len, ..., t-1]
    y: crime w roku t
    """
    df_train = df_train.sort_values("year").reset_index(drop=True)
    exo = df_train[exo_cols].values.astype(np.float32)
    y = df_train["crime"].values.astype(np.float32)

    X_list, y_list = [], []
    for i in range(seq_len, len(df_train)):
        X_list.append(exo[i - seq_len:i])
        y_list.append(y[i])

    return np.array(X_list), np.array(y_list)


def predict_for_year(df_all_exo_sorted: pd.DataFrame, year: int, x_scaler: StandardScaler,
                     model: tf.keras.Model, y_scaler: StandardScaler, seq_len: int) -> float:
    """
    Predykcja crime dla konkretnego roku na podstawie exo z poprzednich seq_len lat.
    """
    window = df_all_exo_sorted[df_all_exo_sorted["year"] < year].tail(seq_len)
    if len(window) < seq_len:
        return np.nan

    X_in = window[exo_cols].values.astype(np.float32)
    X_in_s = x_scaler.transform(X_in).reshape(1, seq_len, len(exo_cols))

    y_hat_s = model.predict(X_in_s, verbose=0).reshape(-1)[0]
    y_hat = float(y_scaler.inverse_transform([[y_hat_s]])[0, 0])
    return y_hat

all_preds = []
all_metrics = []

for unit in units:
    df_u = df[df["unit"] == unit].copy().sort_values("year")

    df_train = df_u[df_u["year"] <= TRAIN_END].copy()
    df_test = df_u[(df_u["year"] >= TEST_START) & (df_u["year"] <= TEST_END)].copy()

    if len(df_train) < SEQ_LEN + 6 or len(df_test) < 2:
        continue

    X, y = make_train_sequences_exo_to_y(df_train, SEQ_LEN)

    x_scaler = StandardScaler()
    X_s = x_scaler.fit_transform(X.reshape(-1, len(exo_cols))).reshape(X.shape)

    y_scaler = StandardScaler()
    y_s = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

    model = build_lstm(SEQ_LEN, len(exo_cols))

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=12, factor=0.5, min_lr=1e-5)

    model.fit(
        X_s, y_s,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es, rlrop]
    )

    exo_all = df_u[(df_u["year"] <= TEST_END)][["year"] + exo_cols].copy()
    exo_all = exo_all.sort_values("year").reset_index(drop=True)

    preds = []
    for yr in df_test["year"].values:
        y_hat = predict_for_year(exo_all, int(yr), x_scaler, model, y_scaler, SEQ_LEN)

        cap = float(df_train["crime"].max() * 1.8)
        y_hat = float(np.clip(y_hat, 0.0, cap))

        preds.append(y_hat)

    df_out = df_test[["year", "unit", "crime"]].copy()
    df_out["predicted_crime"] = preds

    valid = df_out["predicted_crime"].notna()
    y_true = df_out.loc[valid, "crime"].values
    y_pred = df_out.loc[valid, "predicted_crime"].values

    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-6))) * 100
    else:
        r2, mae, rmse, mape = np.nan, np.nan, np.nan, np.nan

    for col, val in [("R2", r2), ("MAE", mae), ("RMSE", rmse), ("MAPE", mape)]:
        df_out[col] = val
    df_out["Model"] = "LSTM_exo_only"

    all_preds.append(df_out)
    all_metrics.append({"unit": unit, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape, "Model": "LSTM_exo_only"})

    plt.figure(figsize=(10, 5))
    plt.plot(df_train["year"], df_train["crime"], marker="o", label=f"Dane historyczne (do {TRAIN_END})")
    plt.plot(df_test["year"], df_test["crime"], marker="o", label=f"Rzeczywiste dane {TEST_START}-{TEST_END}")
    plt.plot(df_test["year"], df_out["predicted_crime"], marker="s", linestyle="--",
             label=f"Predykcja LSTM {TEST_START}-{TEST_END}")

    plt.title(f"Predykcja przestępczości (LSTM) - {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości / 1000 mieszkańców")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_LSTM_2024.png", dpi=200)
    plt.close()

if all_preds:
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)

    metrics_df = pd.DataFrame(all_metrics).sort_values("R2", ascending=False)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    print("Gotowe!")
    print(f"- predykcje: {OUTPUT_PREDICTIONS}")
    print(f"- metryki:   {OUTPUT_METRICS}")
    print(f"- wykresy:   {OUTPUT_PLOTS}")
else:
    print("Brak wyników - za mało danych po filtrowaniu.")
