"""
ProphetForecaster
-------------------
Thin wrapper around Meta's Prophet. Decomposes the series into
trend + yearly + weekly seasonality, with built-in holiday support if needed.
"""

import pandas as pd
from prophet import Prophet


class ProphetForecaster:
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True,
                 daily_seasonality: bool = False):
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
        )
        self._history = None

    def fit(self, df: pd.DataFrame):
        """df must have columns ["date", "sales"]."""
        self._history = df.reset_index(drop=True).copy()
        train_df = self._history.rename(columns={"date": "ds", "sales": "y"})
        self.model.fit(train_df)
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        out = forecast[["ds", "yhat"]].tail(periods).reset_index(drop=True)
        return out.rename(columns={"ds": "date"})
