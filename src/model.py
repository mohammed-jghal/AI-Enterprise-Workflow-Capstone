"""Baseline, ARIMA and SARIMA forecasting with reproducible evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import pickle, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

MODEL_VERSION='1.0.0'

def metrics(actual, predicted):
    a=np.asarray(actual,dtype=float); p=np.asarray(predicted,dtype=float)
    err=a-p
    return {'rmse':float(np.sqrt(np.mean(err**2))), 'mae':float(np.mean(np.abs(err))), 'mape':float(np.mean(np.abs(err)/np.maximum(np.abs(a),1e-8))*100)}

def baseline_forecast(train, horizon, window=7):
    value=float(np.mean(np.asarray(train)[-window:]))
    return np.repeat(value,horizon)

def fit_forecaster(train, kind):
    if kind=='baseline': return {'kind':kind,'train':np.asarray(train,dtype=float)}
    if kind=='arima': return {'kind':kind,'fit':ARIMA(np.asarray(train,dtype=float), order=(2,1,2)).fit()}
    if kind=='sarima': return {'kind':kind,'fit':SARIMAX(np.asarray(train,dtype=float), order=(2,1,2), seasonal_order=(1,0,1,7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)}
    raise ValueError('Unknown model type')

def forecast(fitted, horizon):
    if fitted['kind']=='baseline': return baseline_forecast(fitted['train'],horizon)
    return np.asarray(fitted['fit'].forecast(horizon), dtype=float)

def evaluate_series(series, test_days=30):
    if isinstance(getattr(series, 'index', None), pd.DatetimeIndex):
        # A country with no revenue on a day is absent from the grouped
        # file; without reindexing, "30 days" would mean 30 observations
        # on irregular spacing and ARIMA would assume daily steps.
        series = series.asfreq('D', fill_value=0)
    index=getattr(series, 'index', None)
    y=np.asarray(series,dtype=float); split=len(y)-test_days; train,test=y[:split],y[split:]
    train_start=str(index[0].date()) if index is not None and hasattr(index[0], 'date') else ''
    train_end=str(index[split-1].date()) if index is not None and hasattr(index[split-1], 'date') else ''
    test_start=str(index[split].date()) if index is not None and hasattr(index[split], 'date') else ''
    test_end=str(index[-1].date()) if index is not None and hasattr(index[-1], 'date') else ''
    out=[]
    for kind in ['baseline','arima','sarima']:
        # A failed fit must be reported as a failure, never silently
        # replaced by baseline numbers labelled with the model's name.
        try:
            pred=forecast(fit_forecaster(train,kind),len(test))
            m=metrics(test,pred)
            out.append({'model':kind.upper(),'train_start':train_start,'train_end':train_end,'test_start':test_start,'test_end':test_end,**m})
        except Exception as exc:
            out.append({'model':kind.upper(),'train_start':train_start,'train_end':train_end,'test_start':test_start,'test_end':test_end,
                        'rmse':None,'mae':None,'mape':None,'error':f'{type(exc).__name__}: {exc}'})
    base=out[0]['rmse']
    for row in out:
        row['improvement_vs_baseline_pct']=float((base-row['rmse'])/base*100) if base and row['rmse'] is not None else None
    return out

def train_and_save(series, artifact='models/revenue_model.pkl'):
    y=np.asarray(series,dtype=float)
    fitted=fit_forecaster(y,'sarima')
    payload={'model_version':MODEL_VERSION,'trained_at':datetime.now(timezone.utc).isoformat(),'kind':'sarima','fitted':fitted,'order':(2,1,2),'seasonal_order':(1,0,1,7)}
    path=Path(artifact); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f: pickle.dump(payload,f)
    return payload

def load_model(artifact='models/revenue_model.pkl'):
    with Path(artifact).open('rb') as f: return pickle.load(f)
