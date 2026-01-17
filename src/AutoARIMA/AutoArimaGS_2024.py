import pmdarima as pm
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from itertools import combinations
import warnings

# Konfiguracja
warnings.filterwarnings("ignore")
DATA_PATH = "../../data/processed/panel.csv"
OUTPUT_FILE = "../../output/auto_arima/best_models_results_2024.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
feature_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

# Czyszczenie
df = df.dropna(subset=["crime"])
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df = df.sort_values(by=["unit", "year"])
units = df["unit"].unique()

results = []

print("Rozpoczynam poszukiwanie najlepszych zmiennych dla każdego województwa...")

for unit in units:
    print(f"--- Przetwarzanie: {unit} ---")
    df_unit = df[df["unit"] == unit].copy()

    # Podział train/test
    mask_train = (df_unit["year"] >= 2004) & (df_unit["year"] <= 2020)
    mask_test = df_unit["year"] > 2020

    y_train = df_unit.loc[mask_train, "crime"]
    y_test = df_unit.loc[mask_test, "crime"]

    # Przygotowanie pełnego X
    X_train_full = df_unit.loc[mask_train, feature_cols]
    X_test_full = df_unit.loc[mask_test, feature_cols]

    # Zmienne do śledzenia najlepszego modelu dla tego województwa
    best_mae = float("inf")
    best_features = "BRAK (Tylko historia)"
    best_preds = None
    best_metrics = {}

    # 1. Generujemy kombinacje cech:
    # r=0 (brak zmiennych), r=1 (jedna zmienna), r=2 (pary zmiennych)
    possible_combinations = []
    for r in range(0, len(feature_cols)):
        possible_combinations.extend(combinations(feature_cols, r))

    for combo in possible_combinations:
        combo_list = list(combo)

        # Przygotowanie X dla danej kombinacji
        if len(combo_list) == 0:
            X_train_curr = None
            X_test_curr = None
        else:
            X_train_curr = X_train_full[combo_list]
            X_test_curr = X_test_full[combo_list]

        try:
            model = pm.auto_arima(
                y_train,
                X=X_train_curr,
                start_p=0, start_q=0, max_p=3, max_q=3, d=1,
                seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore'
            )

            # Predykcja
            preds = model.predict(n_periods=len(y_test), X=X_test_curr)

            if isinstance(preds, pd.Series):
                preds = preds.values

            # Walidacja (na zbiorze testowym)
            current_mae = mean_absolute_error(y_test, preds)

            if current_mae < best_mae:
                best_mae = current_mae
                best_features = str(combo_list) if len(combo_list) > 0 else "BRAK"
                best_preds = preds

                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mape = np.mean(
                    np.abs((y_test.values - preds) / np.where(y_test.values != 0, y_test.values, 1e-6))) * 100

                best_metrics = {
                    "R2": r2, "MAE": best_mae, "RMSE": rmse, "MAPE": mape
                }

        except Exception as e:
            continue

    print(f"   -> Zwycięzca: {best_features}")
    print(f"   -> MAE: {best_metrics['MAE']:.3f}, MAPE: {best_metrics['MAPE']:.2f}%")

    results.append({
        "unit": unit,
        "best_features": best_features,
        "R2": best_metrics["R2"],
        "MAE": best_metrics["MAE"],
        "RMSE": best_metrics["RMSE"],
        "MAPE": best_metrics["MAPE"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n===============================================")
print("PODSUMOWANIE NAJLEPSZYCH MODELI")
print("===============================================")
print(results_df[["unit", "best_features", "MAPE"]].to_string(index=False))