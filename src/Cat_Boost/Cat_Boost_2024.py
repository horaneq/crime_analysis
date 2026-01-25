import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor


DATA_PATH = "data/processed/panel.csv"
OUTPUT_PLOTS = "output/catboost/forecast2024/"
OUTPUT_PREDICTIONS = "output/catboost/forecast_catboost_2024.csv"
os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)

SEED = 42
ITERATIONS = 3000
LEARNING_RATE = 0.03
DEPTH = 6
EARLY_STOPPING = 200

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

df = pd.read_csv(DATA_PATH)

feature_cols_all = [
    "unemployment", "population_density", "avg_salary",
    "public_safety_exp", "education_exp", "migration_balance",
    "tourism_usage", "inflation"
]

df = df.dropna(subset=["crime"]).copy()
df[feature_cols_all] = df[feature_cols_all].fillna(df[feature_cols_all].median())
df = df.sort_values(by=["unit", "year"]).reset_index(drop=True)

all_final_predictions = []
final_metrics = []

print("Generowanie finalnych modeli CatBoost (per województwo) z najlepszymi cechami...\n")

for unit, features in best_configs.items():
    print(f"Przetwarzanie: {unit} (Cechy: {features})")

    df_unit = df[df["unit"] == unit].copy()

    mask_train = (df_unit["year"] >= 2004) & (df_unit["year"] <= 2020)
    mask_test = (df_unit["year"] >= 2021) & (df_unit["year"] <= 2024)

    y_train = df_unit.loc[mask_train, "crime"].astype(float).values
    y_test = df_unit.loc[mask_test, "crime"].astype(float).values

    if len(y_train) < 8 or len(y_test) == 0:
        print(f"  - pomijam (za mało danych train/test)")
        continue

    if len(features) == 0:
        preds = np.full(len(y_test), y_train[-1], dtype=float)
    else:
        X_train = df_unit.loc[mask_train, features].astype(float)
        X_test = df_unit.loc[mask_test, features].astype(float)

        try:
            model = CatBoostRegressor(
                loss_function="RMSE",
                iterations=ITERATIONS,
                learning_rate=LEARNING_RATE,
                depth=DEPTH,
                random_seed=SEED,
                verbose=False
            )

            n_train = len(X_train)
            split = int(np.floor(n_train * 0.8))
            if split < 5 or (n_train - split) < 3:
                model.fit(X_train, y_train)
            else:
                X_tr, y_tr = X_train.iloc[:split], y_train[:split]
                X_ev, y_ev = X_train.iloc[split:], y_train[split:]

                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_ev, y_ev),
                    use_best_model=True,
                    early_stopping_rounds=EARLY_STOPPING
                )

            preds = model.predict(X_test).reshape(-1)
            preds = np.maximum(preds, 0.0)

        except Exception as e:
            print(f"  - Błąd dla {unit}: {e}")
            preds = np.full(len(y_test), y_train[-1], dtype=float)

    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mape = float(np.mean(np.abs((y_test - preds) / np.where(y_test != 0, y_test, 1e-6))) * 100)
    r2 = r2_score(y_test, preds) if len(y_test) >= 2 else np.nan

    df_res = df_unit.loc[mask_test].copy()
    df_res["predicted_crime"] = preds
    df_res["used_features"] = str(features)
    all_final_predictions.append(df_res)

    final_metrics.append({
        "unit": unit, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Features": str(features)
    })

    plt.figure(figsize=(10, 5))

    plt.plot(df_unit.loc[mask_train, "year"], df_unit.loc[mask_train, "crime"],
             label="Dane historyczne (2004-2020)", marker="o")

    plt.plot(df_unit.loc[mask_test, "year"], y_test,
             label="Dane rzeczywiste (2021-2024)", color="green", marker="o")

    plt.plot(df_unit.loc[mask_test, "year"], preds,
             label="Predykcja CatBoost (2021-2024)", color="red", linestyle="--", marker="s")

    plt.title(f"{unit}\nModel: CatBoost, RMSE:{rmse:.2f} , MAE:{mae:.2f}, MAPE:{mape:.2f}%, R2:{r2:.2f}")
    plt.xlabel("Rok")
    plt.ylabel("Wskaźnik przestępczości")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PLOTS}/{unit}_final.png", dpi=200)
    plt.close()

pred_df = pd.concat(all_final_predictions, ignore_index=True)
pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)

metrics_df = pd.DataFrame(final_metrics)

print("\nGotowe!")
print(f"- wykresy: {OUTPUT_PLOTS}")
print(f"- predykcje: {OUTPUT_PREDICTIONS}")
print(metrics_df[["unit", "RMSE", "MAE", "MAPE", "R2"]].round(2))
OUT_METRICS = "output/catboost/metrics_catboost_local_2021_2024.csv"
os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
metrics_df.to_csv(OUT_METRICS, index=False)
print(f"- metryki: {OUT_METRICS}")
