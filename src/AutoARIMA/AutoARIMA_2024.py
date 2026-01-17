import pmdarima as pm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- KONFIGURACJA ---
DATA_PATH = "../../data/processed/panel.csv"
OUTPUT_PLOTS = "../../output/auto_arima/forecast2024/"
OUTPUT_PREDICTIONS = "../../output/auto_arima/forecast_auto_arima_2024.csv"
os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)

# --- Najlepsze konfiguracje zmiennych dla każdego z województw ---
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

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
feature_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

df = df.dropna(subset=["crime"])
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df = df.sort_values(by=["unit", "year"])

all_final_predictions = []
final_metrics = []

print("Generowanie finalnych modeli z najlepszymi parametrami...\n")

for unit, features in best_configs.items():
    print(f"Przetwarzanie: {unit} (Cechy: {features})")

    df_unit = df[df["unit"] == unit].copy()

    # Podział train/test (2004-2020 vs 2021+)
    mask_train = (df_unit["year"] >= 2004) & (df_unit["year"] <= 2020)
    mask_test = df_unit["year"] > 2020

    y_train = df_unit.loc[mask_train, "crime"]
    y_test = df_unit.loc[mask_test, "crime"]

    # Przygotowanie X (tylko wybrane cechy)
    if len(features) > 0:
        X_train = df_unit.loc[mask_train, features]
        X_test = df_unit.loc[mask_test, features]
    else:
        X_train = None
        X_test = None

    try:
        model = pm.auto_arima(
            y_train, X=X_train,
            start_p=0, start_q=0, max_p=3, max_q=3, d=1,
            seasonal=False, stepwise=True,
            suppress_warnings=True, error_action='ignore'
        )

        # Predykcja
        preds = model.predict(n_periods=len(y_test), X=X_test)
        if isinstance(preds, pd.Series): preds = preds.values

    except Exception as e:
        print(f"Błąd dla {unit}: {e}")
        preds = np.full(len(y_test), y_train.iloc[-1])

    # Metryki
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test.values - preds) / np.where(y_test.values != 0, y_test.values, 1e-6))) * 100
    r2 = r2_score(y_test, preds)

    # Zapis danych
    df_res = df_unit.loc[mask_test].copy()
    df_res["predicted_crime"] = preds
    df_res["used_features"] = str(features)
    all_final_predictions.append(df_res)

    final_metrics.append({
        "unit": unit, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Features": str(features)
    })

    # --- WYKRES ---
    plt.figure(figsize=(10, 5))

    # Historia
    plt.plot(df_unit.loc[mask_train, "year"], y_train, label="Dane historyczne (2004-2020)", marker="o")

    # Rzeczywistość testowa
    plt.plot(df_unit.loc[mask_test, "year"], y_test, label="Dane rzeczywiste (2021-2024)", color="green", marker="o")

    # Predykcja
    plt.plot(df_unit.loc[mask_test, "year"], preds, label="Predykcja AutoARIMA (2021-2024)", color="red", linestyle="--", marker="s")

    plt.title(f"{unit}\nModel: AutoARIMA, RMSE:{rmse:.2f} , MAE:{mae:.2f}, MAPE:{mape:.2f}%, R2:{r2:.2f}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_final.png", dpi=200)
    plt.close()

# Zapis CSV
pd.concat(all_final_predictions).to_csv(OUTPUT_PREDICTIONS, index=False)
metrics_df = pd.DataFrame(final_metrics)
print("\nGotowe!")
print(metrics_df[["unit", "RMSE", "MAE", "MAPE", "R2"]].round(2))