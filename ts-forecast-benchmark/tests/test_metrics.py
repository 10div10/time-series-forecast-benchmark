import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import mae, rmse, mape, evaluate


def test_mae_perfect_prediction():
    y = [10, 20, 30]
    assert mae(y, y) == 0.0


def test_rmse_perfect_prediction():
    y = [10, 20, 30]
    assert rmse(y, y) == 0.0


def test_mape_known_value():
    y_true = [100, 200]
    y_pred = [110, 180]
    # errors: 10% and 10%
    assert round(mape(y_true, y_pred), 2) == 10.0


def test_mae_simple_case():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    assert round(mae(y_true, y_pred), 2) == round((2 + 2 + 3) / 3, 2)


def test_evaluate_returns_all_keys():
    y_true = [10, 20, 30]
    y_pred = [11, 19, 31]
    result = evaluate(y_true, y_pred)
    assert set(result.keys()) == {"MAE", "RMSE", "MAPE (%)"}
