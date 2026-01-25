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
OUTPUT_PLOTS = "output/lstm/forecast2024/"
OUTPUT_PREDICTIONS = "output/lstm/forecast_lstm_2024.csv"
OUTPUT_METRICS = "output/lstm/metrics_lstm_2024.csv"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

TRAIN_START = 2004
TRAIN_END = 2020
TEST_START = 2021
TEST_END = 2024

SEQ_LEN = 5
EPOCHS = 600
BATCH_SIZE = 8

best_configs = {
    'Dolnośląskie': ['unemployment', 'population_density', 'migration_balance', 'inflation'],
    'Kujawsko-Pomorskie': ['unemployment', 'population_density', 'migration_balance', 'tourism_usage'],
    'Lubelskie': ['unemployment', 'population_density', 'public_safety_exp', 'education_exp', 'inflation'],
    'Lubuskie': ['public_safety_exp', 'tourism_usage'],
    'Mazowieckie': ['unemployment', 'tourism_usage', 'inflation'],
    'Małopolskie': ['unemployment', 'tourism_usage'],
    'Opolskie': ['unemployment', 'population_density', 'migration_balance', 'tourism_usage'],
    'Podkarpackie': ['avg_salary', 'public_safety_exp', 'migration_balance'],
    'Podlaskie': ['unemployment', 'migration_balance', 'tourism_usage', 'inflation'],
    'Pomorskie': ['unemployment', 'public_safety_exp', 'tourism_usage'],
    'Warmińsko-Mazurskie': ['population_density', 'avg_salary', 'migration_balance'],
    'Wielkopolskie': ['unemployment', 'avg_salary', 'education_exp', 'migration_balance', 'inflation'],
    'Zachodniopomorskie': ['unemployment', 'population_density', 'avg_salary', 'education_exp', 'inflation'],
    'Łódzkie': ['unemployment', 'public_safety_exp', 'tourism_usage'],
    'Śląskie': ['unemployment', 'avg_salary', 'public_safety_exp', 'inflation'],
    'Świętokrzyskie': ['unemployment', 'avg_salary', 'migration_balance', 'inflation']
}

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) > 1e-6, y_true, 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

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

def make_sequences_exo_to_y(df_unit, exo_cols, seq_len):
    df_u = df_unit.sort_values("year").reset_index(drop=True)

    exo = df_u[exo_cols].values.astype(np.float32)
    y = df_u["crime"].values.astype(np.float32)
    years = df_u["year"].values.astype(int)

    X_list, y_list, years_list = [], [], []

    for i in range(seq_len, len(df_u)):
        X_list.append(exo[i - seq_len:i])
        y_list.append(y[i])
        years_list.append(years[i])

    return np.array(X_list), np.array(y_list), np.array(years_list)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["crime"]).copy()

all_feature_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]
df[all_feature_cols] = df[all_feature_cols].fillna(df[all_feature_cols].median())
df = df.sort_values(by=["unit", "year"])

all_preds = []
all_metrics = []

print("Trenowanie i predykcja LSTM per-województwo (jak ARIMA)...\n")

for unit, exo_cols in best_configs.items():
    print(f"Przetwarzanie: {unit} (Cechy: {exo_cols})")

    df_unit = df[df["unit"] == unit].copy()

    X_all, y_all, years_all = make_sequences_exo_to_y(df_unit, exo_cols, SEQ_LEN)

    train_mask = (years_all >= TRAIN_START) & (years_all <= TRAIN_END)
    test_mask = (years_all >= TEST_START) & (years_all <= TEST_END)

    if train_mask.sum() < 6 or test_mask.sum() < 1:
        print(f"  Pomijam {unit} - za mało próbek train/test.\n")
        continue

    X_train, y_train, years_train = X_all[train_mask], y_all[train_mask], years_all[train_mask]
    X_test, y_test, years_test = X_all[test_mask], y_all[test_mask], years_all[test_mask]

    x_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, len(exo_cols)))
    X_train_s = x_scaler.transform(X_train.reshape(-1, len(exo_cols))).reshape(X_train.shape)
    X_test_s = x_scaler.transform(X_test.reshape(-1, len(exo_cols))).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    model = build_lstm((SEQ_LEN, len(exo_cols)))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=25, factor=0.5, min_lr=1e-5)

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
    y_pred = np.maximum(y_pred, 0.0)

    mae = mean_absolute_error(y_test, y_pred)
    _rmse = rmse(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan

    all_metrics.append({
        "unit": unit,
        "RMSE": _rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Features": str(exo_cols),
        "Model": "LSTM_exo_only",
        "train_years": f"{TRAIN_START}-{TRAIN_END}",
        "test_years": f"{TEST_START}-{TEST_END}",
        "seq_len": SEQ_LEN
    })

    df_pred = pd.DataFrame({
        "unit": unit,
        "year": years_test.astype(int),
        "actual_crime": y_test.astype(float),
        "predicted_crime": y_pred.astype(float),
        "used_features": str(exo_cols),
        "Model": "LSTM_exo_only"
    }).sort_values("year")

    all_preds.append(df_pred)

    df_hist = df_unit[(df_unit["year"] >= TRAIN_START) & (df_unit["year"] <= TRAIN_END)].copy()

    plt.figure(figsize=(10, 5))
    plt.plot(df_hist["year"], df_hist["crime"], label=f"Dane historyczne ({TRAIN_START}-{TRAIN_END})", marker="o")
    plt.plot(df_pred["year"], df_pred["actual_crime"], label=f"Dane rzeczywiste ({TEST_START}-{TEST_END})", marker="o")
    plt.plot(df_pred["year"], df_pred["predicted_crime"], label=f"Predykcja LSTM ({TEST_START}-{TEST_END})", linestyle="--", marker="s")

    plt.title(f"{unit}\nModel: LSTM, RMSE:{_rmse:.2f}, MAE:{mae:.2f}, MAPE:{mape:.2f}%, R2:{r2:.2f}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_LSTM_final.png", dpi=200)
    plt.close()

    print(f"  RMSE={_rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%, R2={r2:.3f}\n")

preds_df = pd.concat(all_preds, ignore_index=True) if len(all_preds) else pd.DataFrame()
metrics_df = pd.DataFrame(all_metrics)

preds_df.to_csv(OUTPUT_PREDICTIONS, index=False)
metrics_df.to_csv(OUTPUT_METRICS, index=False)

print("Gotowe!")
print(f"- wykresy: {OUTPUT_PLOTS}")
print(f"- predykcje: {OUTPUT_PREDICTIONS}")
print(f"- metryki: {OUTPUT_METRICS}")
print(metrics_df[["unit", "RMSE", "MAE", "MAPE", "R2"]].round(2))
