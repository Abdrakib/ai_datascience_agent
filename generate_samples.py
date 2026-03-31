"""
generate_samples.py — creates sample CSV datasets for testing the agent.
Run once: python generate_samples.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "datasets"

# ── Sample 1: Classification — patient readmission ────────────────────────────
n = 500
df_clf = pd.DataFrame({
    "age":            rng.integers(20, 85, n),
    "bmi":            rng.normal(27, 5, n).round(1),
    "blood_pressure": rng.integers(60, 140, n),
    "glucose":        rng.normal(100, 25, n).round(1),
    "num_medications": rng.integers(0, 12, n),
    "days_in_hospital": rng.integers(1, 14, n),
    "gender":         rng.choice(["Male", "Female"], n),
    "smoker":         rng.choice(["Yes", "No"], n),
    "insurance":      rng.choice(["Private", "Medicare", "Medicaid", "None"], n),
})
# Introduce some missing values (realistic)
for col in ["bmi", "glucose", "blood_pressure"]:
    mask = rng.random(n) < 0.07
    df_clf.loc[mask, col] = np.nan

# Target: readmitted (logistic-ish probability)
log_odds = (
    -3
    + 0.03 * df_clf["age"].fillna(df_clf["age"].mean())
    + 0.04 * df_clf["num_medications"]
    + 0.08 * df_clf["days_in_hospital"]
    + 0.5  * (df_clf["smoker"] == "Yes").astype(int)
)
prob = 1 / (1 + np.exp(-log_odds))
df_clf["readmitted"] = (rng.random(n) < prob).astype(int)

df_clf.to_csv(OUT / "sample_healthcare_classification.csv", index=False)
print(f"Saved: sample_healthcare_classification.csv  ({len(df_clf)} rows)")

# ── Sample 2: Regression — house price ───────────────────────────────────────
n = 600
sqft        = rng.integers(600, 4500, n)
bedrooms    = rng.integers(1, 6, n)
bathrooms   = rng.choice([1, 1.5, 2, 2.5, 3, 3.5], n)
age_years   = rng.integers(0, 60, n)
garage      = rng.choice([0, 1, 2], n)
neighborhood = rng.choice(["Urban", "Suburban", "Rural"], n)
condition   = rng.choice(["Excellent", "Good", "Fair", "Poor"], n)

price = (
    50000
    + sqft * 110
    + bedrooms * 8000
    + bathrooms * 12000
    - age_years * 500
    + garage * 15000
    + np.where(neighborhood == "Urban", 40000,
       np.where(neighborhood == "Suburban", 20000, 0))
    + np.where(condition == "Excellent", 30000,
       np.where(condition == "Good", 10000,
        np.where(condition == "Poor", -20000, 0)))
    + rng.normal(0, 20000, n)
).astype(int)

df_reg = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "garage_spaces": garage,
    "neighborhood": neighborhood,
    "condition": condition,
    "price": price,
})

# Introduce some missing values
for col in ["age_years", "garage_spaces"]:
    mask = rng.random(n) < 0.05
    df_reg.loc[mask, col] = np.nan

df_reg.to_csv(OUT / "sample_housing_regression.csv", index=False)
print(f"Saved: sample_housing_regression.csv  ({len(df_reg)} rows)")

print("\nDone. Both sample datasets are in the datasets/ folder.")
