import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
import os
import warnings

# Ignorowanie ostrzeżeń dla czystości outputu
warnings.filterwarnings("ignore")

# --- KONFIGURACJA ---
DATA_PATH = "../../data/processed/panel.csv"
OUTPUT_DIR = "../../output/auto_arima/forecast/"
OUTPUT_PREDICTIONS = "../../output/auto_arima/forecast_arima.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
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

# --- 1. WCZYTANIE I PRZYGOTOWANIE DANYCH ---
print("Wczytywanie danych...")
df = pd.read_csv(DATA_PATH)

feature_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]


df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df = df.dropna(subset=["crime"])
df = df.sort_values(by=["unit", "year"])


future_years = [2025, 2026, 2027, 2028, 2029]
n_future = len(future_years)

all_future_forecasts = []

print(f"Rozpoczynam prognozowanie na lata {future_years}...\n")

for unit, features in best_configs.items():
    print(f"--- {unit} ---")

    df_unit = df[df["unit"] == unit].copy()

    # Trenujemy na CAŁEJ historii (do 2024 włącznie)
    mask_history = df_unit["year"] >= 2004
    df_history = df_unit.loc[mask_history]

    y_history = df_history["crime"]
    X_history = df_history[features] if features else None

    # =========================================================
    # KROK 1: Prognoza zmiennych egzogenicznych (Gospodarka)
    # =========================================================
    X_future = pd.DataFrame(index=range(n_future))


    if features:
        for feature in features:
            try:
                # Prognozujemy wskaźnik gospodarczy na podstawie danych 2004-2024
                feat_series = df_history[feature]
                feat_model = pm.auto_arima(feat_series, seasonal=False,
                                           error_action='ignore', suppress_warnings=True)
                feat_pred = feat_model.predict(n_periods=n_future)
                X_future[feature] = feat_pred.values if isinstance(feat_pred, pd.Series) else feat_pred
            except:
                X_future[feature] = df_history[feature].iloc[-1]
    else:
        X_future = None

    # =========================================================
    # KROK 2: Prognoza Główna (Przestępczość)
    # =========================================================
    try:
        model = pm.auto_arima(
            y_history,
            X=X_history,
            start_p=0, start_q=0, max_p=3, max_q=3, d=1,
            seasonal=False, stepwise=True,
            suppress_warnings=True, error_action='ignore'
        )

        # Prognozujemy przyszłość, używając przewidzianych zmiennych gospodarczych
        forecast, conf_int = model.predict(n_periods=n_future, X=X_future, return_conf_int=True)

        # Formatowanie wyników
        if isinstance(forecast, pd.Series): forecast = forecast.values

    except Exception as e:
        print(f"   Błąd krytyczny modelu: {e}")
        forecast = np.full(n_future, y_history.iloc[-1])
        conf_int = np.array([[x, x] for x in forecast])

    # Zabezpieczenie przed ujemną przestępczością
    forecast = np.maximum(forecast, 0)
    conf_int = np.maximum(conf_int, 0)

    # Zapis do tabeli
    future_df = pd.DataFrame({
        "unit": unit,
        "year": future_years,
        "predicted_crime": forecast,
        "lower_bound": conf_int[:, 0],  # Dolna granica optymistyczna
        "upper_bound": conf_int[:, 1]  # Górna granica pesymistyczna
    })
    all_future_forecasts.append(future_df)

    # =========================================================
    # WYKRES (Historia + Przyszłość)
    # =========================================================
    plt.figure(figsize=(12, 6))

    # Historia
    plt.plot(df_history["year"], y_history, label="Dane historyczne (2004-2024)", marker="o")

    # Prognoza
    plt.plot(future_years, forecast, label="PROGNOZA (2025-2029)", color="red", marker="s", linestyle="--")

    # Przedział ufności
    plt.fill_between(future_years, conf_int[:, 0], conf_int[:, 1], color='red', alpha=0.15,
                     label="Przedział ufności 95%")

    plt.title(
        f"Prognoza przestępczości: {unit} (2025-2029)")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{unit}_AutoARIMA.png", dpi=200)
    plt.close()

# Zapis zbiorczy CSV
final_df = pd.concat(all_future_forecasts)
csv_path = f"{OUTPUT_DIR}/full_forecast_2025_2029.csv"
final_df.to_csv(csv_path, index=False)

print(f"\nGotowe! Wyniki zapisano w: {csv_path}")
print(f"Wykresy zapisano w katalogu: {OUTPUT_DIR}")