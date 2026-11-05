# AI Enterprise Workflow - Capstone Project

Capstone project for the IBM AI Enterprise Workflow Specialization on Coursera.

## Problem

Predict future revenue for a streaming media company, total and by country, for the next 30 days by default.

## Project structure

```
data/input/   # invoices JSON files (Nov 2017 - Dec 2019)
nb/           # analysis.ipynb (EDA), results.ipynb (model comparison)
src/          # ingest.py, model.py, app.py, log.py, monitor.py, config.py
tests/        # app_test.py, model_test.py, log_test.py
Dockerfile
requirements.txt
run_app.py
run_tests.py
```

## Endpoints

- `POST /predict?date={date}&duration={duration}&country={country}`
- `POST /logs?type={type}` where type is `ingest`, `train` or `predict`

## Usage

```bash
python run_app.py
python run_tests.py
curl --request POST http://127.0.0.1/predict?date=2018-11-20
```

```bash
docker build -t app .
docker run -it --rm -p 3000:80 --name app app
```

## Marking criteria checklist

- [ ] Unit tests for the API
- [ ] Unit tests for the model
- [ ] Unit tests for logging
- [ ] All unit tests run with a single script and pass
- [ ] Mechanism to monitor performance
- [ ] Isolation of read/write unit tests from production models and logs
- [ ] API works for a specific country and for all countries
- [ ] Data ingestion as a function/script for automation
- [ ] Multiple models compared (ARIMA vs SARIMA)
- [ ] EDA uses visualizations
- [ ] Everything containerized in a working Docker image
- [ ] Visualization comparing the model to the baseline

## Status

Work in progress. Implementation is built step by step by the author.
