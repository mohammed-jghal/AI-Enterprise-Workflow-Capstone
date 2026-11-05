"""API tests.

Self-contained by design: the trained model artifact (models/*.pkl) is
gitignored and absent from a fresh clone, so the fixture trains a model
into a temporary directory and points the app at temporary data and log
paths. The tests never read or write the repository's models/ or logs/,
and they pass on a fresh clone without running `make train` first.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.app import create_app
from src.model import train_and_save


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    for name in ("3 revenue_country.csv", "4 revenue_total.csv"):
        shutil.copy(ROOT / "data" / "output" / name, tmp / name)
    total = pd.read_csv(tmp / "4 revenue_total.csv", parse_dates=["invoice_date"])
    model_path = tmp / "revenue_model.pkl"
    train_and_save(total["revenue"], str(model_path))
    app = create_app(test_config={
        "MODEL_PATH": model_path,
        "COUNTRY_PATH": tmp / "3 revenue_country.csv",
        "TOTAL_PATH": tmp / "4 revenue_total.csv",
        "LOG_DIR": tmp / "logs",
    })
    app.config["TESTING"] = True
    return app.test_client()


def test_total_prediction(client):
    r = client.post('/predict?date=2018-11-20')
    assert r.status_code == 200 and r.json['country'] == 'TOTAL'


def test_country_prediction(client):
    r = client.post('/predict?date=2018-11-20&country=US&duration=5')
    assert r.status_code == 200 and len(r.json['daily_predictions']) == 5


def test_invalid_date(client):
    assert client.post('/predict?date=not-a-date').status_code == 400


def test_invalid_country(client):
    assert client.post('/predict?date=2018-11-20&country=ZZ').status_code == 400


def test_health_ready(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json['status'] == 'ready'


def test_request_id_is_echoed(client):
    r = client.post('/predict?date=2018-11-20', headers={'X-Request-ID': 'test-request-1'})
    assert r.status_code == 200 and r.json['request_id'] == 'test-request-1' \
        and r.headers['X-Request-ID'] == 'test-request-1'


def test_invalid_duration_has_error_envelope(client):
    r = client.post('/predict?date=2018-11-20&duration=0')
    assert r.status_code == 400 and r.json['error']['code'] == 'invalid_duration'


def test_version_endpoint(client):
    r = client.get('/version')
    assert r.status_code == 200 and 'model_version' in r.json


def test_predictions_vary_with_sarima(client):
    r = client.post('/predict?date=2018-11-20&duration=5')
    assert r.status_code == 200
    preds = r.json['daily_predictions']
    assert len(preds) == 5
    assert r.json['forecast_method'] == 'sarima_refit'
    assert len(set(preds)) > 1, "SARIMA forecast must not be a flat constant"


def test_country_predictions_vary_with_sarima(client):
    r = client.post('/predict?date=2018-11-20&country=US&duration=5')
    assert r.status_code == 200
    preds = r.json['daily_predictions']
    assert len(preds) == 5
    assert r.json['forecast_method'] == 'sarima_refit'
    assert len(set(preds)) > 1, "SARIMA forecast must not be a flat constant"


def test_unknown_route_returns_404_not_500(client):
    # The error handler used to convert every exception (including 404s)
    # into a 500; unknown routes must stay 404.
    assert client.get('/no-such-route').status_code == 404


def test_predict_date_after_data_end_returns_400(client):
    # The data ends 2019-12-31. A forecast requested from 2020-06-01
    # would be anchored at 2020-01-01 but labelled June: reject it.
    r = client.post('/predict?date=2020-06-01')
    assert r.status_code == 400
    assert r.json['error']['code'] == 'date_out_of_range'


def test_predict_date_before_history_returns_400(client):
    r = client.post('/predict?date=2017-01-01')
    assert r.status_code == 400
    assert r.json['error']['code'] == 'date_out_of_range'


def test_predict_without_date_defaults_to_first_forecastable_day(client):
    # No date must not mean "today" (years past the data): the first
    # forecastable day is the day after the last available date.
    r = client.post('/predict?duration=5')
    assert r.status_code == 200
    assert r.json['start_date'] == '2020-01-01'
    assert r.json['forecast_method'] == 'sarima_refit'
