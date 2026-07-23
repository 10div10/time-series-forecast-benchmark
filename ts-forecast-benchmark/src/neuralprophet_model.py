"""
NeuralProphetForecaster
------------------------
Wraps NeuralProphet (https://neuralprophet.com/) in the same interface as
ArimaForecaster / ProphetForecaster so it drops straight into compare_models.py.

NeuralProphet extends Prophet's decomposable trend + seasonality model with a
small AR-Net (autoregressive neural network) component, so it can pick up
short-term autocorrelation patterns that vanilla Prophet's smooth trend/season
decomposition misses -- while still remaining interpretable via component plots.

Install:
    pip install neuralprophet

Expected input format: a DataFrame with columns ["date", "sales"], matching
data/sales_data.csv, same as ProphetForecaster.
"""

import pandas as pd
import torch

# --- Compatibility shim ---
# PyTorch 2.6 flipped torch.load's default from weights_only=False to True,
# which breaks NeuralProphet's internal checkpoint loading (it pickles config
# objects, not just tensors). NeuralProphet's checkpoints are created locally
# by this same process, so restoring the pre-2.6 default here is safe.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from neuralprophet import NeuralProphet, set_log_level

# Quiet down NeuralProphet's fairly verbose training logs by default.
set_log_level("ERROR")


class NeuralProphetForecaster:
    def __init__(
        self,
        n_lags: int = 14,
        n_forecasts: int = 1,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        epochs: int = 100,
        learning_rate: float = None,
    ):
        """
        Parameters
        ----------
        n_lags : int
            Size of the AR-Net autoregressive window (how many past days feed
            the neural component). 0 disables AR-Net and NeuralProphet behaves
            like Prophet. 7-14 is a reasonable starting point for daily retail
            data with weekly structure.
        n_forecasts : int
            Number of steps ahead predicted per forward pass. Keep at 1 for a
            simple apples-to-apples comparison against ARIMA/Prophet's
            single-step-style evaluation loop; raise it for true multi-step
            forecasting.
        epochs : int
            NeuralProphet auto-selects a learning rate and epoch count if left
            at defaults; override here if training looks unstable.
        """
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.model = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        self._history_df = None

    def fit(self, df: pd.DataFrame):
        """
        df must have columns ["date", "sales"]. Internally renamed to
        NeuralProphet's required ["ds", "y"] schema (same convention as Prophet).
        """
        train_df = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]].copy()
        train_df["ds"] = pd.to_datetime(train_df["ds"])
        self._history_df = train_df
        self.metrics_ = self.model.fit(train_df, freq="D")
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        """
        Forecast `periods` steps beyond the training data.
        Returns a DataFrame with columns ["date", "yhat"] to match
        ProphetForecaster's output shape for compare_models.py.

        NOTE: for a clean single multi-step forecast in one forward pass,
        construct this class with n_forecasts == periods (compare_models.py
        does this). If n_forecasts < periods, NeuralProphet only produces
        n_forecasts steps and the remainder are forward-filled with the last
        predicted value -- called out in the README as a limitation rather
        than hidden.
        """
        if self._history_df is None:
            raise RuntimeError("Call .fit() before .predict().")

        future = self.model.make_future_dataframe(
            self._history_df, periods=periods, n_historic_predictions=False
        )
        forecast = self.model.predict(future)

        # NeuralProphet stores multi-step forecasts on a diagonal: yhatK at
        # row (origin_index + K) is the K-step-ahead prediction made from a
        # single origin at the end of training. So the full forecast for
        # that one origin is read off the anti-diagonal, not a single row.
        n_forecasts = self.n_forecasts
        preds = []
        for k in range(1, n_forecasts + 1):
            row_idx = len(forecast) - n_forecasts + (k - 1)
            preds.append(forecast.iloc[row_idx][f"yhat{k}"])

        last_date = self._history_df["ds"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(preds), freq=freq)
        out = pd.DataFrame({"date": future_dates, "yhat": preds})

        if len(out) < periods:
            pad_dates = pd.date_range(out["date"].iloc[-1] + pd.Timedelta(days=1),
                                       periods=periods - len(out), freq=freq)
            pad = pd.DataFrame({"date": pad_dates, "yhat": out["yhat"].iloc[-1]})
            out = pd.concat([out, pad], ignore_index=True)

        return out.head(periods).reset_index(drop=True)

    def plot_components(self):
        """Optional: NeuralProphet's component decomposition plot (trend/seasonality/AR),
        handy for a 'model interpretability' section in the README."""
        if self._history_df is None:
            raise RuntimeError("Call .fit() before .plot_components().")
        future = self.model.make_future_dataframe(self._history_df, periods=0)
        forecast = self.model.predict(future)
        return self.model.plot_components(forecast)
