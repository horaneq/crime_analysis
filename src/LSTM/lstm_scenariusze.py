import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
layers = tf.keras.layers
callbacks = tf.keras.callbacks
models = tf.keras.models


DATA_PATH = "data/processed/panel.csv"
OUTPUT_DIR = "output/lstm/features/"
OUTPUT_CSV = "output/lstm/features/feature_importances_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 5
EPOCHS = 200
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
feature_cols = ["crime"] + exo_cols

units = sorted(df["unit"].unique())

def make_sequences(df_unit, seq_len):
    df_unit = df_unit.sort_values("year").reset_index(drop=True)
    X_list, y_list, years_list = [], [], []
    values = df_unit[feature_cols].values.astype(np.float32)
    target = df_unit["crime"].values.astype(np.float32)
    years = df_unit["year"].values
    for i in range(seq_len, len(df_unit)):
        X_list.append(values[i-seq_len:i])
        y_list.append(target[i])
        years_list.append(years[i])
    return np.array(X_list), np.array(y_list), np.array(years_list)

def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

all_rows = []

for unit in units:
    df_unit = df[df["unit"] == unit].copy()

    X, y, years = make_sequences(df_unit, SEQ_LEN)
    if len(X) == 0:
        continue

    train_mask = (years <= 2020)
    test_mask = (years > 2020)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) < 5 or len(X_test) == 0:
        continue

    n_features = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    X_train_s = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    model = build_lstm((SEQ_LEN, n_features))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[es])

    base_pred = model.predict(X_test_s, verbose=0).reshape(-1)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))

    for j, col in enumerate(feature_cols):
        X_perm = X_test_s.copy()
        flat = X_perm[:, :, j].reshape(-1)
        np.random.shuffle(flat)
        X_perm[:, :, j] = flat.reshape(X_perm[:, :, j].shape)

        pred_perm = model.predict(X_perm, verbose=0).reshape(-1)
        rmse_perm = np.sqrt(mean_squared_error(y_test, pred_perm))

        importance = rmse_perm - base_rmse 
        all_rows.append({
            "unit": unit,
            "feature": col,
            "importance_rmse_increase": importance,
            "base_rmse": base_rmse
        })

imp_df = pd.DataFrame(all_rows)
imp_df.to_csv(OUTPUT_CSV, index=False)

print("Gotowe! Zapisano permutation importance dla LSTM.")
