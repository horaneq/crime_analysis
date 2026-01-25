import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow as tf
layers = tf.keras.layers
callbacks = tf.keras.callbacks
models = tf.keras.models

DATA_PATH = "../../data/processed/panel.csv"
SINGLE_SCENARIOS_OUTPUT = "../../output/lstm/feature_influence/"
OUTPUT_PREDICTIONS = "../../output/lstm/forecast_lstm_feature_influence.csv"

os.makedirs(SINGLE_SCENARIOS_OUTPUT, exist_ok=True)

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
future_years = [2025, 2026, 2027, 2028, 2029]
factors = [0.8, 1.2]

def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

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

all_predictions = []

for unit in units:
    df_unit = df[df["unit"] == unit].copy().sort_values("year")
    df_train = df_unit[df_unit["year"] <= 2024].copy()
    if len(df_train) < SEQ_LEN + 3:
        continue

    last_exo = df_train[exo_cols].iloc[-1].to_dict()
    future_base = pd.DataFrame([{**last_exo, "year": y} for y in future_years])

    X, y, _ = make_sequences(df_train, SEQ_LEN)
    n_features = X.shape[-1]

    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, n_features))
    X_s = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)

    model = build_lstm((SEQ_LEN, n_features))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(X_s, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[es])

    def iterative_forecast(future_exo_df):
        history = df_train[feature_cols].copy().reset_index(drop=True)
        preds = []
        for i, row in future_exo_df.iterrows():
            X_in = history.tail(SEQ_LEN).values.astype(np.float32)
            X_in_s = scaler.transform(X_in.reshape(-1, n_features)).reshape(1, SEQ_LEN, n_features)
            y_hat = float(max(model.predict(X_in_s, verbose=0).reshape(-1)[0], 0))
            preds.append(y_hat)
            new_row = {"crime": y_hat, **{c: row[c] for c in exo_cols}}
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        return np.array(preds)

    y_base = iterative_forecast(future_base)
    df_base_out = pd.DataFrame({
        "year": future_years,
        "unit": unit,
        "predicted_crime": y_base,
        "scenario": "Base_2024_stable",
        "Model": "LSTM"
    })
    all_predictions.append(df_base_out)

    for feature in exo_cols:
        plt.figure(figsize=(12, 6))
        plt.plot(df_train["year"], df_train["crime"], marker="s", color="black", label="Historyczne dane")
        plt.plot(future_years, y_base, marker="o", color="gray", label="Prognoza bazowa (stabilne 2024)")

        for factor in factors:
            future_scen = future_base.copy()
            future_scen[feature] = future_scen[feature] * factor

            y_scen = iterative_forecast(future_scen)

            scen_name = f"{feature}_x{factor:.1f}"
            df_scen_out = pd.DataFrame({
                "year": future_years,
                "unit": unit,
                "predicted_crime": y_scen,
                "scenario": scen_name,
                "Model": "LSTM"
            })
            all_predictions.append(df_scen_out)

            label_text = f"{feature} x{factor:.1f}"
            plt.plot(future_years, y_scen, linestyle="--", marker="^" if factor > 1 else "v", label=label_text)

        plt.title(f"Prognoza przestępczości (LSTM) - {unit} - wpływ: {feature}")
        plt.xlabel("Rok")
        plt.ylabel("Wskaźnik przestępczości")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, loc="best")
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(f"{SINGLE_SCENARIOS_OUTPUT}/{unit}_influence_{feature}.png", dpi=200)
        plt.close()

pred_df = pd.concat(all_predictions, ignore_index=True)
pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)

print("Gotowe! Wygenerowano scenariusze LSTM i wykresy wpływu cech.")
