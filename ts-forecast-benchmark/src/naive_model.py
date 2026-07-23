"""
SeasonalNaiveForecaster
------------------------
The forecast every other model in this repo has to beat. Predicts each future
value as the actual value observed exactly one seasonal period ago (default
7 days, i.e. "sales today = sales same weekday last week").

If your ARIMA/Prophet/NeuralProphet/LightGBM models can't beat this, that's a
useful (and honest) finding, not a bug.
"""

import pandas as pd


class SeasonalNaiveForecaster:
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self._history = None

    def fit(self, df: pd.DataFrame):
        """df must have columns ["date", "sales"]."""
        self._history = df.reset_index(drop=True).copy()
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError("Call .fit() before .predict().")

        last_season = self._history["sales"].values[-self.season_length:]
        reps = (periods // self.season_length) + 1
        preds = list(last_season) * reps
        preds = preds[:periods]

        last_date = self._history["date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

        return pd.DataFrame({"date": future_dates, "yhat": preds})
