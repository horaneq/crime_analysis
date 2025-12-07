import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ============================================================
#   ŚCIEŻKI I KONFIGURACJA
# ============================================================
DATA_PATH = "../../data/processed/panel.csv"
# Nowy folder do zapisu WYKRESÓW dla pojedynczej cechy
SINGLE_SCENARIOS_OUTPUT = "../../output/random_forest/feature_influence"
OUTPUT_PREDICTIONS = "../../output/random_forest/forecast_rf_feature_influence.csv"

os.makedirs(SINGLE_SCENARIOS_OUTPUT, exist_ok=True)

# ============================================================
#   PRZYGOTOWANIE DANYCH
# ============================================================
df = pd.read_csv(DATA_PATH)

feature_cols = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]


df = df.dropna(subset=["crime"])

units = df["unit"].unique()
future_years = [2025, 2026, 2027, 2028, 2029]

all_predictions = []
all_importances = []


# ============================================================
#   FUNKCJA DO SYMULACJI SCENARIUSZY JEDNEGO WSKAŹNIKA
# ============================================================
def apply_scenario_single_feature(future_df, feature, factor):
    """
    Zmienia tylko jedną metrykę w przyszłych danych o dany współczynnik.
    """
    df_scenario = future_df.copy()

    if feature in df_scenario.columns:
        df_scenario[feature] *= factor
    return df_scenario


# ============================================================
#   MODELOWANIE I PREDYKCJE Z WAŻNOŚCIĄ CECH
# ============================================================

for unit in units:
    df_unit = df[df["unit"] == unit].copy()

    # Imputacja: Uzupełnianie braków medianą TYLKO dla bieżącej jednostki
    unit_medians = df_unit[feature_cols].median()
    df_unit[feature_cols] = df_unit[feature_cols].fillna(unit_medians)

    df_train = df_unit[df_unit["year"] <= 2024]

    X_train = df_train[feature_cols]
    y_train = df_train["crime"]

    # Inicjalizacja i uczenie Random Forest
    crime_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=5
    )
    crime_model.fit(X_train, y_train)

    # ------------------------------
    # 1. OBLICZENIE I ZAPIS WAŻNOŚCI CECH
    # ------------------------------
    importances = crime_model.feature_importances_

    importance_df = pd.DataFrame({
        'unit': unit,
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    all_importances.append(importance_df)

    # ------------------------------
    # 2. Prognoza bazowa i scenariuszowa
    # ------------------------------

    # Przygotowanie danych przyszłych (wszystkie cechy równe wartościom z 2024 r.)
    last_row = df_train[feature_cols].iloc[-1]
    future_df_base = pd.DataFrame(
        [last_row.values for _ in future_years],
        columns=feature_cols
    )
    future_df_base["year"] = future_years
    future_df_base["unit"] = unit

    # === A. Prognoza bazowa (Base Scenario) ===
    # Obliczamy prognozę bazową
    df_base = future_df_base.copy()
    y_baseline = crime_model.predict(df_base[feature_cols])
    df_base["predicted_crime"] = np.maximum(y_baseline, 0)
    df_base["scenario"] = "Base_Scenario_Unchanged"
    all_predictions.append(df_base)

    factors = [0.8, 1.2]

    # === B. Pętla generująca ODDZIELNE wykresy dla każdej cechy ===
    for feature in feature_cols:

        # 1. Inicjalizacja wykresu dla DANEJ CECHY
        plt.figure(figsize=(12, 6))

        # Rysowanie danych historycznych
        plt.plot(df_train["year"], df_train["crime"], marker="s", color="black",
                 label="Historyczne dane")

        # Rysowanie Prognozy Bazowej
        plt.plot(df_base["year"], df_base["predicted_crime"], linestyle="-", marker="o",
                 color="gray", label="Prognoza bazowa (Stabilne 2024)")

        for factor in factors:
            scen_name = f"{feature}_x{factor:.1f}"

            # 2. Zastosowanie scenariusza (tylko na jednej cechy)
            df_scen = apply_scenario_single_feature(future_df_base.copy(), feature, factor)

            y_scen = crime_model.predict(df_scen[feature_cols])
            df_scen["predicted_crime"] = np.maximum(y_scen, 0)
            df_scen["scenario"] = scen_name

            all_predictions.append(df_scen)

            # 3. Rysowanie Scenariusza
            if factor > 1.0:
                color = 'red'
                marker = '^'
                label_text = f"Wzrost {feature} o 20% (x1.2)"
            else:
                color = 'green'
                marker = 'v'
                label_text = f"Spadek {feature} o 20% (x0.8)"

            plt.plot(df_scen["year"], df_scen["predicted_crime"], linestyle="--", marker=marker,
                     alpha=1.0, color=color, label=label_text)

        # 4. Zapis wykresu
        plt.title(f"Prognoza przestępczości (RF) w {unit} - Wpływ: {feature}")
        plt.xlabel("Rok")
        plt.ylabel("Wskaźnik przestępczości")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, loc='best')
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(f"{SINGLE_SCENARIOS_OUTPUT}/{unit}_influence_{feature}.png", dpi=200)
        plt.close()

# ============================================================
#   ZAPIS WYNIKÓW KOŃCOWYCH
# ============================================================

predictions_df = pd.concat(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

print("\n-----------------------------------------------------------")
print("✅ Gotowe!")
print("Wygenerowano prognozy scenariuszowe oraz wykresy ważności cech.")
print(f"Prognozy scenariuszowe (jeden wykres na cechę) zapisano w: {SINGLE_SCENARIOS_OUTPUT}")
print("-----------------------------------------------------------\n")