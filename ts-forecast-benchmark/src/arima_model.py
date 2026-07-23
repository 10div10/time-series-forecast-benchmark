"""
ArimaForecaster
-----------------
Uses pmdarima's auto_arima to search (p,d,q)(P,D,Q,m) and statsmodels under
the hood for fitting/forecasting. seasonal_period=7 for daily data with
weekly patterns; use 12 for monthly data with yearly patterns.
"""

import pandas as pd
import pmdarima as pm


class ArimaForecaster:
    def __init__(self, seasonal_period: int = 7, seasonal: bool = True, suppress_warnings: bool = True):
        self.seasonal_period = seasonal_period
        self.seasonal = seasonal
        self.suppress_warnings = suppress_warnings
        self.model = None
        self._history = None

    def fit(self, df: pd.DataFrame):
        """df must have columns ["date", "sales"]."""
        self._history = df.reset_index(drop=True).copy()
        series = self._history["sales"]

        self.model = pm.auto_arima(
            series,
            seasonal=self.seasonal,
            m=self.seasonal_period,
            suppress_warnings=self.suppress_warnings,
            error_action="ignore",
            stepwise=True,
        )
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict().")

        forecast = self.model.predict(n_periods=periods)
        last_date = self._history["date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

        return pd.DataFrame({"date": future_dates, "yhat": forecast})
