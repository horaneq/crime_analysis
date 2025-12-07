import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Random Forest dla głównej predykcji

# --- Stałe konfiguracyjne ---
DATA_PATH = "../../data/processed/panel.csv"
OUTPUT_PLOTS = "../../output/random_forest/forecast/"
OUTPUT_PREDICTIONS = "../../output/random_forest/forecast_rf.csv"

# Tworzenie katalogu wyjściowego
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
future_years = [2025, 2026, 2027, 2028, 2029]

all_predictions = []


# ============================================================
#   ETAP I: PREDYKCJA CECH (Autoregresja Liniowa)
# ============================================================

def forecast_feature_linear(values, years, future_years):
    """
    Uczy Regresję Liniową (rok -> cecha) do ekstrapolacji cech na przyszłe lata.
    """
    model = LinearRegression()
    X = np.array(years).reshape(-1, 1)
    y = np.array(values)

    model.fit(X, y)

    future_pred = model.predict(np.array(future_years).reshape(-1, 1))
    return future_pred


# ============================================================
#   ETAP II: MODELOWANIE GŁÓWNE (Random Forest Regressor)
# ============================================================

for unit in units:

    df_unit = df[df["unit"] == unit].copy()

    # ------------------------------
    # 1. Prognoza cech na lata 2025-2029 (LR)
    # ------------------------------
    future_df = pd.DataFrame({"year": future_years})

    for col in feature_cols:
        past_values = df_unit[col].values
        years = df_unit["year"].values

        # Używamy LR do prognozowania cech
        future_vals = forecast_feature_linear(past_values, years, future_years)
        future_df[col] = future_vals

    # ------------------------------
    # 2. Model regresji dla 'crime' (RANDOM FOREST)
    # ------------------------------
    X_train = df_unit[feature_cols]
    y_train = df_unit["crime"]

    # Inicjalizacja Random Forest Regressor
    crime_model_rf = RandomForestRegressor(
        n_estimators=100,  # Liczba drzew
        random_state=42,  # Ziarno losowości
        max_depth=5,  # Kontrola złożoności
        min_samples_split=5
    )

    crime_model_rf.fit(X_train, y_train)

    # Predykcja na podstawie prognozowanych cech
    future_df["predicted_crime"] = crime_model_rf.predict(future_df[feature_cols])

    # Minimalny poziom – >= 0
    future_df["predicted_crime"] = np.maximum(future_df["predicted_crime"], 0)
    future_df["unit"] = unit

    all_predictions.append(future_df)

    # ------------------------------
    # 3. Wykres dla województwa
    # ------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df_unit["year"], df_unit["crime"], marker="o",
             label="Historyczne dane")

    plt.plot(future_df["year"], future_df["predicted_crime"],
             marker="s", linestyle="--", label="Prognoza (RF) 2025–2039")

    plt.title(f"Trend przestępczości – {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_RF.png", dpi=200)
    plt.close()

# ============================================================
#   ZAPIS WYNIKÓW
# ============================================================

predictions_df = pd.concat(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

print("Gotowe! Wygenerowano prognozy przestępczości (Random Forest) z użyciem prognoz cech (Regresja Liniowa).")