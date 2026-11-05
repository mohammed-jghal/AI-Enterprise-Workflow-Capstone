import numpy as np
from src.model import metrics, baseline_forecast, fit_forecaster, forecast

def test_metrics_zero_error(): assert metrics([1,2],[1,2])['rmse']==0

def test_baseline_shape(): assert len(baseline_forecast([1,2,3],4))==4

def test_arima_forecast_shape():
    y=np.arange(80,dtype=float)+np.sin(np.arange(80)); assert len(forecast(fit_forecaster(y,'arima'),5))==5

def test_sarima_forecast_shape():
    y=np.arange(80,dtype=float)+np.sin(np.arange(80)); assert len(forecast(fit_forecaster(y,'sarima'),5))==5
