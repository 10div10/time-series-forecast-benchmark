"""
Trains all six forecasters on the same train/test split, evaluates each on
the held-out test period, prints a metrics table, and saves a comparison
plot + metrics.csv to outputs/.

Models: SeasonalNaive, ETS, ARIMA, Prophet, NeuralProphet, LightGBM
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from naive_model import SeasonalNaiveForecaster
from ets_model import ETSForecaster
from arima_model import ArimaForecaster
from prophet_model import ProphetForecaster
from neuralprophet_model import NeuralProphetForecaster
from lightgbm_model import LightGBMForecaster
from metrics import evaluate

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TEST_DAYS = 30


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def train_test_split(df: pd.DataFrame, test_days: int):
    train = df.iloc[:-test_days].reset_index(drop=True)
    test = df.iloc[-test_days:].reset_index(drop=True)
    return train, test


def run_model(name, forecaster, train_df, test_df):
    print(f"Training {name}...")
    forecaster.fit(train_df)
    forecast = forecaster.predict(periods=len(test_df))
    scores = evaluate(test_df["sales"].values, forecast["yhat"].values)
    scores["model"] = name
    return scores, forecast


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train_df, test_df = train_test_split(df, TEST_DAYS)

    models = {
        "SeasonalNaive": SeasonalNaiveForecaster(season_length=7),
        "ETS": ETSForecaster(seasonal_periods=7),
        "ARIMA": ArimaForecaster(seasonal_period=7),
        "Prophet": ProphetForecaster(),
        "NeuralProphet": NeuralProphetForecaster(n_lags=14, n_forecasts=TEST_DAYS, epochs=50),
        "LightGBM": LightGBMForecaster(),
    }

    results = []
    forecasts = {}
    for name, model in models.items():
        try:
            scores, forecast = run_model(name, model, train_df, test_df)
            results.append(scores)
            forecasts[name] = forecast
        except Exception as e:
            print(f"  {name} failed: {e}")

    results_df = pd.DataFrame(results).set_index("model")
    results_df = results_df[["MAE", "RMSE", "MAPE (%)"]].sort_values("MAE")
    print("\n=== Results (sorted by MAE) ===")
    print(results_df.to_string())
    results_df.to_csv(OUTPUT_DIR / "metrics.csv")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_df["date"].tail(90), train_df["sales"].tail(90),
            color="#4b5563", lw=1.2, label="Historical (train)")
    ax.plot(test_df["date"], test_df["sales"], color="#111827", lw=2, label="Actual (test)")

    colors = ["#e07a5f", "#3d5a80", "#81b29a", "#f2cc8f", "#9d4edd"]
    for (name, forecast), color in zip(forecasts.items(), colors):
        ax.plot(forecast["date"], forecast["yhat"], lw=1.8, ls="--", label=name, color=color)

    ax.set_title("Time Series Forecast Benchmark: 6 Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sales")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_plot.png", dpi=160, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_DIR / 'comparison_plot.png'}")
    print(f"Saved metrics to {OUTPUT_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
