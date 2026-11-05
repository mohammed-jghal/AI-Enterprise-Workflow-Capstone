import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.app import app

def test_total_prediction():
    c=app.test_client(); r=c.post('/predict?date=2018-11-20'); assert r.status_code==200 and r.json['country']=='TOTAL'
def test_country_prediction():
    c=app.test_client(); r=c.post('/predict?date=2018-11-20&country=US&duration=5'); assert r.status_code==200 and len(r.json['daily_predictions'])==5
def test_invalid_date():
    c=app.test_client(); assert c.post('/predict?date=not-a-date').status_code==400
def test_invalid_country():
    c=app.test_client(); assert c.post('/predict?date=2018-11-20&country=ZZ').status_code==400

def test_health_ready():
    c=app.test_client(); r=c.get('/health'); assert r.status_code==200 and r.json['status']=='ready'

def test_request_id_is_echoed():
    c=app.test_client(); r=c.post('/predict?date=2018-11-20', headers={'X-Request-ID':'test-request-1'})
    assert r.status_code==200 and r.json['request_id']=='test-request-1' and r.headers['X-Request-ID']=='test-request-1'

def test_invalid_duration_has_error_envelope():
    c=app.test_client(); r=c.post('/predict?date=2018-11-20&duration=0')
    assert r.status_code==400 and r.json['error']['code']=='invalid_duration'

def test_version_endpoint():
    c=app.test_client(); r=c.get('/version'); assert r.status_code==200 and 'model_version' in r.json

def test_predictions_vary_with_sarima():
    c=app.test_client(); r=c.post('/predict?date=2018-11-20&duration=5')
    assert r.status_code==200
    preds=r.json['daily_predictions']
    assert len(preds)==5
    assert r.json['forecast_method']=='sarima_refit'
    assert len(set(preds))>1, "SARIMA forecast must not be a flat constant"

def test_country_predictions_vary_with_sarima():
    c=app.test_client(); r=c.post('/predict?date=2018-11-20&country=US&duration=5')
    assert r.status_code==200
    preds=r.json['daily_predictions']
    assert len(preds)==5
    assert r.json['forecast_method']=='sarima_refit'
    assert len(set(preds))>1, "SARIMA forecast must not be a flat constant"
