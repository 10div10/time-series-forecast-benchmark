"""
LightGBMForecaster
--------------------
Reframes forecasting as tabular regression: lag features + rolling stats +
calendar features -> LightGBM regressor. Forecasts recursively (each
predicted value feeds back in as a lag for the next step), which is the
standard approach for multi-step tree-based forecasting.

This is the "ML" counterpoint to the statistical models (ARIMA/ETS) and the
decomposition models (Prophet/NeuralProphet) in this repo -- reuses the same
XGBoost/LightGBM feature-engineering pattern from the two-tower recommender
and visa-risk-scoring projects.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


class LightGBMForecaster:
    def __init__(self, lags=(1, 2, 3, 7, 14), rolling_windows=(7, 14),
                 n_estimators: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 5, random_state: int = 42):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=-1,
        )
        self._history = None
        self.feature_cols_ = None

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["dayofweek"] = out["date"].dt.dayofweek
        out["month"] = out["date"].dt.month
        out["day"] = out["date"].dt.day

        for lag in self.lags:
            out[f"lag_{lag}"] = out["sales"].shift(lag)
        for w in self.rolling_windows:
            out[f"rolling_mean_{w}"] = out["sales"].shift(1).rolling(w).mean()
            out[f"rolling_std_{w}"] = out["sales"].shift(1).rolling(w).std()

        return out

    def fit(self, df: pd.DataFrame):
        """df must have columns ["date", "sales"], sorted ascending by date."""
        self._history = df.reset_index(drop=True).copy()
        self._history["date"] = pd.to_datetime(self._history["date"])

        feat_df = self._make_features(self._history).dropna().reset_index(drop=True)
        self.feature_cols_ = [c for c in feat_df.columns if c not in ("date", "sales")]

        X = feat_df[self.feature_cols_]
        y = feat_df["sales"]
        self.model.fit(X, y)
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """
        Recursive multi-step forecast: predict one day at a time, append the
        prediction to the working history, recompute lag/rolling features,
        repeat. Slower than a single vectorized call but keeps lag features
        honest for multi-step horizons.
        """
        if self.feature_cols_ is None:
            raise RuntimeError("Call .fit() before .predict().")

        working = self._history.copy()
        last_date = working["date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

        preds = []
        for d in future_dates:
            working = pd.concat(
                [working, pd.DataFrame({"date": [d], "sales": [np.nan]})],
                ignore_index=True,
            )
            feat_row = self._make_features(working).iloc[[-1]][self.feature_cols_]
            yhat = self.model.predict(feat_row)[0]
            working.loc[working.index[-1], "sales"] = yhat
            preds.append(yhat)

        return pd.DataFrame({"date": future_dates, "yhat": preds})
