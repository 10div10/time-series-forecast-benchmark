"""
Generates a synthetic daily sales series with trend + weekly + yearly
seasonality + noise, so the project runs end-to-end with zero external
data dependencies. Swap data/sales_data.csv with real data whenever ready
(keep the two columns: date, sales).
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


def generate_sales_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 200 + 0.25 * t
    weekly = 15 * np.sin(2 * np.pi * t / 7)
    yearly = 30 * np.sin(2 * np.pi * t / 365)
    noise = rng.normal(0, 8, n_days)

    sales = trend + weekly + yearly + noise
    sales = np.clip(sales, a_min=0, a_max=None).round(2)

    return pd.DataFrame({"date": dates, "sales": sales})


if __name__ == "__main__":
    df = generate_sales_data()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
