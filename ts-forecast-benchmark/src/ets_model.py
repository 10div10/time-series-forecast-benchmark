"""
ETSForecaster
--------------
Holt-Winters exponential smoothing (statsmodels), with additive trend and
additive weekly seasonality by default. Cheap to train, no external deps
beyond statsmodels, and a strong classical baseline that often rivals ARIMA
on short daily series.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ETSForecaster:
    def __init__(self, seasonal_periods: int = 7, trend: str = "add", seasonal: str = "add"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self._history = None

    def fit(self, df: pd.DataFrame):
        """df must have columns ["date", "sales"]."""
        self._history = df.reset_index(drop=True).copy()
        series = self._history.set_index("date")["sales"]
        series.index.freq = "D"

        self.model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit()
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict().")

        forecast = self.model.forecast(periods)
        last_date = self._history["date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

        return pd.DataFrame({"date": future_dates, "yhat": forecast.values})
