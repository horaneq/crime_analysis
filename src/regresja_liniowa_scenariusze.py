import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA_PATH = "../data/processed/panel.csv"
OUTPUT_PLOTS = "../output/plots_scenarios_single_feature/"
OUTPUT_PREDICTIONS = "../output/predictions_scenarios_single_feature.csv"

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
#   FUNKCJA DO SYMULACJI SCENARIUSZY JEDNEGO WSKAŹNIKA
# ============================================================
def apply_scenario_single_feature(future_df, feature, factor):
    """
    Zmienia tylko jedną metrykę w przyszłych danych.
    Pozostałe wskaźniki pozostają bez zmian.
    """
    df_scenario = future_df.copy()
    if feature in df_scenario.columns:
        df_scenario[feature] *= factor
    return df_scenario

# ============================================================
#   MODELOWANIE I PREDYKCJE SCENARIUSZOWE
# ============================================================

for unit in units:
    df_unit = df[df["unit"] == unit].copy()

    # Dane treningowe do 2024
    df_train = df_unit[df_unit["year"] <= 2024]

    X_train = df_train[feature_cols]
    y_train = df_train["crime"]

    crime_model = LinearRegression()
    crime_model.fit(X_train, y_train)

    # Tworzymy przyszłe lata z trendem (ostatnie obserwacje)
    last_row = df_train[feature_cols].iloc[-1]
    future_df = pd.DataFrame([last_row.values for _ in future_years], columns=feature_cols)
    future_df["year"] = future_years
    future_df["unit"] = unit

    plt.figure(figsize=(12, 6))
    plt.plot(df_train["year"], df_train["crime"], marker="s", color="black", label="Historyczne dane")

    # ------------------------------
    # Tworzenie scenariuszy dla każdej metryki
    # ------------------------------
    factors = [1.2]  # przykładowe poziomy zmiany
    scenario_count = 0

    for feature in feature_cols:
        for factor in factors:
            scenario_count += 1
            scen_name = f"{feature}_x{factor:.1f}"
            df_scen = apply_scenario_single_feature(future_df, feature, factor)
            df_scen["predicted_crime"] = crime_model.predict(df_scen[feature_cols])
            df_scen["predicted_crime"] = np.maximum(df_scen["predicted_crime"], 0)
            df_scen["scenario"] = scen_name

            all_predictions.append(df_scen)

            plt.plot(df_scen["year"], df_scen["predicted_crime"], linestyle="--", marker="o", label=scen_name)

    plt.title(f"Prognoza przestępczości – {unit} (pojedyncze wskaźniki zmieniane)")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_single_feature_scenarios.png", dpi=200)
    plt.close()

# ============================================================
#   ZAPIS WYNIKÓW
# ============================================================

predictions_df = pd.concat(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

print("Gotowe! Wygenerowano prognozy scenariuszowe pojedynczych wskaźników.")
