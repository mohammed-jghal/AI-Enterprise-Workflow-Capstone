import numpy as np
import pytest
from src.model import metrics, baseline_forecast, fit_forecaster, forecast, evaluate_series

def test_metrics_zero_error(): assert metrics([1,2],[1,2])['rmse']==0

def test_baseline_shape(): assert len(baseline_forecast([1,2,3],4))==4

def test_arima_forecast_shape():
    y=np.arange(80,dtype=float)+np.sin(np.arange(80)); assert len(forecast(fit_forecaster(y,'arima'),5))==5

def test_sarima_forecast_shape():
    y=np.arange(80,dtype=float)+np.sin(np.arange(80)); assert len(forecast(fit_forecaster(y,'sarima'),5))==5

def test_evaluate_series_reports_fit_failure_instead_of_silent_baseline(monkeypatch):
    # A failed SARIMA fit used to be silently replaced by baseline numbers
    # labelled "SARIMA". Now the row reports the failure explicitly.
    import src.model as model_mod
    real_fit = model_mod.fit_forecaster

    def failing_fit(train, kind):
        if kind == 'sarima':
            raise RuntimeError('simulated fit failure')
        return real_fit(train, kind)

    monkeypatch.setattr(model_mod, 'fit_forecaster', failing_fit)
    y = np.arange(80, dtype=float) + np.sin(np.arange(80))
    rows = evaluate_series(y, test_days=30)
    sarima = [r for r in rows if r['model'] == 'SARIMA'][0]
    assert sarima['rmse'] is None
    assert 'simulated fit failure' in sarima['error']
    assert sarima['improvement_vs_baseline_pct'] is None
