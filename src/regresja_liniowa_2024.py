import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "../data/processed/panel.csv"
OUTPUT_PLOTS = "../output/plots2024/"
OUTPUT_PREDICTIONS = "../output/predictions_with_metrics.csv"

os.makedirs(OUTPUT_PLOTS, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

df = df.dropna(subset=["crime"])
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

units = df["unit"].unique()

# Wybieramy lata do uczenia i lata testowe
train_years = df[df["year"] <= 2020]["year"].unique()
test_years = df[df["year"] > 2020]["year"].unique()

all_predictions = []

# ============================================================
#   MODELOWANIE I PREDYKCJE
# ============================================================

for unit in units:
    df_unit = df[df["unit"] == unit].copy()

    # Dane treningowe (do 2020)
    df_train = df_unit[df_unit["year"] <= 2020]
    df_test = df_unit[df_unit["year"] > 2020]

    # ------------------------------
    # 1. Model regresji dla crime
    # ------------------------------
    X_train = df_train[feature_cols]
    y_train = df_train["crime"]

    crime_model = LinearRegression()
    crime_model.fit(X_train, y_train)

    # ------------------------------
    # 2. Predykcja na rzeczywistych wskaźnikach testowych
    # ------------------------------
    X_test = df_test[feature_cols]
    y_test = df_test["crime"]

    y_pred = crime_model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # minimalny poziom 0

    df_pred = df_test.copy()
    df_pred["predicted_crime"] = y_pred

    # ------------------------------
    # 3. Obliczenie metryk
    # ------------------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"Województwo: {unit}")
    print(f"R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%\n")

    df_pred["R2"] = r2
    df_pred["MAE"] = mae
    df_pred["RMSE"] = rmse
    df_pred["MAPE"] = mape

    all_predictions.append(df_pred)

    # ------------------------------
    # 4. Wykres
    # ------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df_train["year"], df_train["crime"], marker="o", label="Dane historyczne (do 2020)")
    plt.plot(df_test["year"], y_test, marker="o", label="Rzeczywiste dane 2021+")
    plt.plot(df_test["year"], y_pred, marker="s", linestyle="--", label="Predykcja 2021+")

    plt.title(f"Trend przestępczości – {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_prediction.png", dpi=200)
    plt.close()

# ============================================================
#   ZAPIS WYNIKÓW
# ============================================================

predictions_df = pd.concat(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

print("Gotowe! Wygenerowano predykcje z metrykami porównania do rzeczywistych danych.")
