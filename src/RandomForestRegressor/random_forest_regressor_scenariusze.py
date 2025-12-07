import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "../../data/processed/panel.csv"
SINGLE_SCENARIOS_OUTPUT = "../../output/random_forest/plots_scenarios_single_feature/"
OUTPUT_PLOTS = "../../output/random_forest/features/"
OUTPUT_PREDICTIONS = "../../output/random_forest/forecast_rf_scenarios_optimized.csv"

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
all_importances = []  # Nowa lista do zbierania ważności cech


# ============================================================
#   FUNKCJA DO SYMULACJI SCENARIUSZY JEDNEGO WSKAŹNIKA (bez zmian)
# ============================================================
def apply_scenario_single_feature(future_df, feature, factor):
    """
    Zmienia tylko jedną metrykę w przyszłych danych.
    """
    df_scenario = future_df.copy()
    if feature in df_scenario.columns:
        df_scenario[feature] *= factor
    return df_scenario


# ============================================================
#   MODELOWANIE I PREDYKCJE SCENARIUSZOWE Z WAŻNOŚCIĄ CECH
# ============================================================

for unit in units:
    df_unit = df[df["unit"] == unit].copy()

    df_train = df_unit[df_unit["year"] <= 2024]

    X_train = df_train[feature_cols]
    y_train = df_train["crime"]

    # Inicjalizacja i uczenie Random Forest (bez zmian)
    crime_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=5
    )
    crime_model.fit(X_train, y_train)

    # ------------------------------
    # 1. OBLICZENIE I WIZUALIZACJA WAŻNOŚCI CECH
    # ------------------------------
    importances = crime_model.feature_importances_

    importance_df = pd.DataFrame({
        'unit': unit,
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    all_importances.append(importance_df)

    # Wykres ważności cech
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'], color='teal')
    plt.xlabel("Ważność cechy (Mean Decrease in Impurity)")
    plt.title(f"Ważność cech dla prognozy przestępczości – {unit}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_feature_importance.png", dpi=200)
    plt.close()

    # ------------------------------
    # 2. Prognoza bazowa i scenariuszowa (bez zmian)
    # ------------------------------
    last_row = df_train[feature_cols].iloc[-1]
    future_df = pd.DataFrame([last_row.values for _ in future_years], columns=feature_cols)
    future_df["year"] = future_years
    future_df["unit"] = unit

    # Prognoza bazowa
    y_baseline = crime_model.predict(future_df[feature_cols])
    df_baseline = future_df.copy()
    df_baseline["predicted_crime"] = np.maximum(y_baseline, 0)
    df_baseline["scenario"] = "BAZA_2024_STABILNE"
    all_predictions.append(df_baseline)

    # Inicjalizacja wykresu scenariuszowego
    plt.figure(figsize=(12, 6))
    plt.plot(df_train["year"], df_train["crime"], marker="s", color="black",
             label="Historyczne dane")

    # Rysowanie linii bazowej
    plt.plot(df_baseline["year"], df_baseline["predicted_crime"],
             linestyle="-", color="gray", linewidth=2,
             label="Prognoza bazowa (Stabilne 2024)")

    factors = [1.2]

    for feature in feature_cols:
        for factor in factors:
            scen_name = f"{feature}_x{factor:.1f}"
            df_scen = apply_scenario_single_feature(future_df, feature, factor)
            df_scen["predicted_crime"] = crime_model.predict(df_scen[feature_cols])
            df_scen["predicted_crime"] = np.maximum(df_scen["predicted_crime"], 0)
            df_scen["scenario"] = scen_name

            all_predictions.append(df_scen)

            plt.plot(df_scen["year"], df_scen["predicted_crime"], linestyle="--", marker="o",
                     alpha=1.0, label=scen_name)

    # Zapis wykresu scenariuszowego
    plt.title(f"Prognoza przestępczości (RF) – {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(f"{SINGLE_SCENARIOS_OUTPUT}/{unit}_rf_scenarios.png", dpi=200)
    plt.close()

# ============================================================
#   ZAPIS WYNIKÓW
# ============================================================

predictions_df = pd.concat(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

# Zapis zbiorczego zestawienia ważności cech do osobnego pliku
all_importances_df = pd.concat(all_importances)
all_importances_df.to_csv(f"{OUTPUT_PLOTS}feature_importances_summary.csv", index=False)

print("Gotowe! Wygenerowano prognozy scenariuszowe oraz wykresy ważności cech.")
print(f"Ważność cech zapisana do: {OUTPUT_PLOTS}feature_importances_summary.csv")