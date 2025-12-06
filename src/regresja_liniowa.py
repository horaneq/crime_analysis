import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA_PATH = "../data/processed/panel.csv"
OUTPUT_PLOTS = "../output/plots/"
OUTPUT_PREDICTIONS = "../output/predictions.csv"

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
#   PREDYKCJA CECH: AUTOREGRESJA LINEARNA
# ============================================================

def forecast_feature(values, years, future_years):
    """
    Uczy regresję liniową cecha(t) → cecha(t+1).
    Zwraca prognozę cechy na przyszłe lata.
    """
    model = LinearRegression()
    X = np.array(years).reshape(-1, 1)
    y = np.array(values)

    model.fit(X, y)

    future_pred = model.predict(np.array(future_years).reshape(-1, 1))
    return future_pred


# ============================================================
#   MODELOWANIE KAŻDEGO WOJEWÓDZTWA
# ============================================================

for unit in units:

    df_unit = df[df["unit"] == unit].copy()

    # ------------------------------
    # 1. Prognoza cech na lata 2025-2029
    # ------------------------------
    future_df = pd.DataFrame({"year": future_years})

    for col in feature_cols:
        past_values = df_unit[col].values
        years = df_unit["year"].values

        future_vals = forecast_feature(past_values, years, future_years)
        future_df[col] = future_vals

    # ------------------------------
    # 2. Model regresji dla crime
    # ------------------------------
    X = df_unit[feature_cols]
    y = df_unit["crime"]

    crime_model = LinearRegression()
    crime_model.fit(X, y)

    future_df["predicted_crime"] = crime_model.predict(future_df[feature_cols])

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
             marker="s", linestyle="--", label="Prognoza 2025–2029")

    plt.title(f"Trend przestępczości – {unit}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_trend.png", dpi=200)
    plt.close()

# ============================================================
#   ZAPIS WYNIKÓW
# ============================================================

predictions_df = pd.concat(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)

print("Gotowe! Wygenerowano prognozy cech i przestępczości.")
