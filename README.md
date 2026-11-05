# Revenue Forecasting Platform

A production-oriented revenue forecasting platform for a streaming media company. The repository demonstrates a complete, auditable workflow: monthly JSON ingestion, data quality controls, exploratory analysis, model comparison, artifact creation, a validated Flask API, structured logs, drift monitoring, tests, Docker packaging, and continuous integration.

## What changed in this production-ready edition

The API now has an application factory, environment-based configuration, request correlation IDs, consistent JSON error envelopes, cached read-only datasets, `/health` and `/version` endpoints, strict date/country/horizon validation, and a Docker healthcheck. The repository also includes a Makefile, CI workflow, Python project configuration, a smoke test, an environment template, and a security policy.

Fixes in this revision: ingestion supports both the synthetic sample schema and the real Coursera schema (full country names, `price`, `year`/`month`/`day`); the API forecasts with the SARIMA model instead of a flat 7-day mean (see "Data and modelling evidence"); both notebooks are executed with embedded plots; the test suite is 21 tests, including schema and non-flat-forecast tests.

## Architecture

```text
Monthly invoice JSON
        |
        v
src/ingest.py  -->  data/output/0..4 CSV products
        |
        v
src/model.py   -->  models/revenue_model.pkl + model_metrics.json
        |
        v
src/app.py     -->  /health /version /predict /logs
        |
        v
JSONL audit logs + Wasserstein drift monitor
```

## Data: synthetic sample plus real Coursera schema support

The files under `data/input/` are a deterministic **synthetic sample** shaped like the course data (26 monthly files, November 2017 – December 2019, 4,754 invoice rows). They exist so the pipeline runs end to end without course credentials.

`src/ingest.py` also supports the **real Coursera schema** from the IBM AI Enterprise Workflow course: records with full country names (e.g. `"United Kingdom"`), `price`, `times_viewed`, and `year` / `month` / `day` fields. Ingestion builds the invoice date from `year`/`month`/`day`, uses `price` as revenue, and maps full country names to ISO codes (`United Kingdom` → `GB`, `United States` → `US`, `France` → `FR`, `Germany` → `DE`, `Canada` → `CA`, `Australia` → `AU`, plus other common names). Both schemas produce the same five outputs, so no downstream code changes are needed.

To run on the real course data, replace the files under `data/input/` with the real monthly `invoices-YYYY-MM.json` files and re-run:

```bash
make train   # runs ingestion, then training and metrics
make test
```

Ingestion is covered by tests for both schemas (`tests/ingest_test.py`). The trained model artifact (`models/revenue_model.pkl`, ~15 MB) and the JSONL logs are generated locally and are **not** committed to this repository (see `.gitignore`); regenerate them with `make train`.

## Run locally

```bash
python3 -m venv .venv
. .venv/bin/activate
make install
make train
make test
PORT=8080 python run_app.py
```

The API is then available at `http://127.0.0.1:8080`. The default forecast is 30 days. Country forecasts use a two-letter code.

```bash
curl -X POST 'http://127.0.0.1:8080/predict?date=2018-11-20'
curl -X POST 'http://127.0.0.1:8080/predict?date=2018-11-20&country=US&duration=7'
curl 'http://127.0.0.1:8080/health'
curl -X POST 'http://127.0.0.1:8080/logs?type=predict'
```

Successful prediction responses include `request_id`, model version, country scope, date range, daily predictions, total prediction, and latency. Validation failures return a stable shape such as `{"error":{"code":"invalid_duration","message":"...","request_id":"..."}}` with HTTP 400.

## Data and modelling evidence

`src/ingest.py` creates five deterministic outputs: raw combined data, cleaned data, engineered features, daily revenue by country, and daily total revenue. It records files read, rows accepted, and rows dropped in the ingestion log.

`nb/analysis.ipynb` provides the time-series plot, seasonal decomposition, ACF/PACF analysis, country comparison, volatility view, and written interpretations. Both notebooks are executed, so the plots and the metrics table are embedded in the committed files. `src/model.py` evaluates a 7-day moving-average baseline, ARIMA(2,1,2), and SARIMA(2,1,2) with seasonal order (1,0,1,7) — a weekly seasonal component — on the final 30-day holdout. The output includes RMSE, MAE, MAPE, improvement versus baseline, and explicit train/test dates for total revenue and each country.

The current reproducible sample run (regenerated with `make train` on the synthetic sample) produced the following total-revenue test metrics:

| Model | RMSE | MAE | MAPE | Improvement vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 205.45 | 158.50 | 7.59% | 0.00% |
| ARIMA | 192.08 | 151.11 | 7.27% | 6.51% |
| SARIMA | 152.31 | 128.60 | 6.35% | 25.86% |

The holdout is `2019-12-02` through `2019-12-31`; training ends on `2019-12-01`. Re-run `make train` after changing the data to regenerate the numbers and artifact. Per-country results are in `data/output/model_metrics.json`; note that ARIMA is worse than the baseline for some individual countries (AU, FR, GB, US), while SARIMA beats the baseline for total revenue and for every country listed there.

The API serves predictions from the SARIMA model: for a requested start date it refits SARIMA(2,1,2)(1,0,1,7) on the history available before that date and forecasts from there, because the saved artifact was trained on data ending 2019-12-01 and cannot answer an arbitrary start date directly. The response includes a `forecast_method` field (`sarima_refit` on success). Only if SARIMA fitting fails does the API fall back to a 7-day moving average, and then `forecast_method` is `baseline_fallback` — a flat fallback is never presented as a model forecast. A test asserts that a 5-day forecast is not a flat constant.

## Operations and monitoring

`src/log.py` writes separate JSONL audit streams for ingestion, training, and prediction. The API adds a correlation ID to every response and prediction log. `src/monitor.py` calculates Wasserstein distance between a reference distribution and a current distribution, then returns the distance, configured threshold, drift decision, and either `continue` or `retrain`.

The model artifact is treated as trusted deployment input. Before a real deployment, put the API behind TLS and authentication, restrict access to logs, add rate limiting, use a model registry with signed artifacts, and configure alerts around latency and drift.

## Tests and quality gates

```bash
make quality
make test
```

The suite contains 21 tests covering the API (including a test that SARIMA predictions are not flat), model behaviour, isolated logging, and ingestion of both the synthetic and the real Coursera data schemas. Tests write to temporary directories and never overwrite production `logs/` or `models/`. Run them with `python run_tests.py` — the last verified run passed 21/21.

Note: the CI workflow file `.github/workflows/ci.yml` is not in this repository yet. The GitHub API integration used for publishing cannot create workflow files (it returns 403 for `.github/workflows/`). Add that file via the GitHub web UI to get automated ingestion, training, notebook validation, compilation, and tests on every push and pull request.

## Docker

```bash
docker build -t revenue-forecast-api .
docker run --rm -p 3000:80 --name revenue-forecast-api revenue-forecast-api
BASE_URL=http://127.0.0.1:3000 ./scripts/smoke_test.sh
```

The image runs as a non-root user and exposes a Docker healthcheck. Docker is intentionally not required for the local test suite. Note: the Docker image was not build-verified in the review environment (no Docker daemon available there); the CI workflow builds and exercises the pipeline on push, and the image should be verified with `docker build` before relying on it for a demo.

## Repository map

- `src/`: ingestion, modelling, API, logging, and monitoring modules.
- `tests/`: API, model, and logging tests.
- `nb/`: EDA and model-comparison notebooks.
- `data/input/`: monthly invoice JSON source files.
- `data/output/`: generated deterministic data products and metrics.
- `models/`: trained artifact.
- `.github/workflows/ci.yml`: automated quality gate (not in the repository yet — see the note in "Tests and quality gates").
- `.env.example`: documented runtime configuration.
- `SECURITY.md`: deployment security expectations.
