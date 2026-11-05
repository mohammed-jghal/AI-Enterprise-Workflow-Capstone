# Revenue Forecasting Platform

A production-oriented revenue forecasting platform for a streaming media company. The repository demonstrates a complete, auditable workflow: monthly JSON ingestion, data quality controls, exploratory analysis, model comparison, artifact creation, a validated Flask API, structured logs, drift monitoring, tests, Docker packaging, and continuous integration.

## What changed in this production-ready edition

The API now has an application factory, environment-based configuration, request correlation IDs, consistent JSON error envelopes, cached read-only datasets, `/health` and `/version` endpoints, strict date/country/horizon validation, and a Docker healthcheck. The repository also includes a Makefile, Python project configuration, a smoke test, an environment template, and a security policy. There is **no CI workflow file in this repository yet** — see "Tests and quality gates".

Fixes in this revision: the shipped sample data is now in the **real Coursera schema** (full country names, `price`, `year`/`month`/`day`, `invoice`, `customer_id`, `stream_id`, `times_viewed`), so ingestion exercises the real-schema path end to end instead of only a legacy schema; the API forecasts with the SARIMA model instead of a flat 7-day mean (see "Data and modelling evidence"); both notebooks are executed with embedded plots; the test suite is 29 tests and passes on a fresh clone with no model artifact present; the Dockerfile has a `.dockerignore` and installs runtime dependencies only. Code defects fixed in this revision: unknown routes returned 500 instead of 404; `/predict` accepted a start date past the end of the data and returned a forecast labelled with the wrong dates; the 15 MB model pickle was re-read on every request; a failed ARIMA/SARIMA fit was silently replaced by baseline numbers under the model's name; per-country series were not reindexed to a daily frequency; the runner scripts failed when run from outside the repository root; the test suite failed on a fresh clone because the model artifact is gitignored; `start.sh` and `scripts/smoke_test.sh` were not executable; `.env` was not gitignored.

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

## Data: deterministic synthetic sample in the real Coursera schema

The files under `data/input/` are a deterministic **synthetic sample in the real Coursera schema** (26 monthly files, November 2017 – December 2019, 4,754 invoice rows, 4,746 accepted, 8 intentionally invalid rows dropped). Each record has the real course fields: `country` (full name, e.g. `"United Kingdom"`), `customer_id`, `invoice`, `price`, `stream_id`, `times_viewed`, and `year` / `month` / `day`. They exist so the pipeline runs end to end without course credentials — **Coursera provides no downloadable invoice files; the course data lives only inside the course lab**, so these are synthetic invoices, not real ones, and the sample is labelled as such wherever it is described.

One simplification is stated plainly: the sample has **one invoice per country per day** (6 invoices/day, price = that country's daily revenue). Real course data has many invoices per day. The sample therefore exercises the schema, the date/country mapping, and the revenue aggregation exactly, but it is not a realistic invoice-volume distribution.

`src/ingest.py` also supports the **legacy synthetic schema** (records with `invoice_date`, two-letter `country`, and `amount`) for backward compatibility. Both schemas produce the same five outputs, so no downstream code changes are needed. Ingestion builds the invoice date from `year`/`month`/`day`, uses `price` as revenue, and maps full country names to ISO codes (`United Kingdom` → `GB`, `United States` → `US`, `France` → `FR`, `Germany` → `DE`, `Canada` → `CA`, `Australia` → `AU`, plus other common names).

The conversion from the previous legacy-schema sample was 1 record → 1 record, order preserved, `price` = old `amount` exactly, so daily revenue totals per country are **identical** to the previous sample. That is why the metrics below are unchanged from the previous revision: the schema changed, the revenue series did not. The conversion is reproducible from git history (the legacy files are at the previous commit); the mapping is: `invoice_date` → `year`/`month`/`day` strings, two-letter code → full name, `amount` → `price`, `invoice_id` → sequential 6-digit `invoice` from 489434, plus deterministic `customer_id` (float, 10% missing), `stream_id`, and `times_viewed` from `random.Random(42)`; the 8 invalid rows became real-schema rows with all fields `None` and are still dropped.

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
make install   # installs runtime + dev/test/notebook dependencies
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

The current reproducible sample run (regenerated with `make train` on the real-schema synthetic sample) produced the following total-revenue test metrics. They are **identical to the previous revision's metrics** because the conversion preserved the daily revenue series exactly (see "Data" above) — the schema changed, the numbers did not, and this was verified by diffing the revenue outputs before and after the conversion.

| Model | RMSE | MAE | MAPE | Improvement vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 205.45 | 158.50 | 7.59% | 0.00% |
| ARIMA | 192.08 | 151.11 | 7.27% | 6.51% |
| SARIMA | 152.31 | 128.60 | 6.35% | 25.86% |

The holdout is `2019-12-02` through `2019-12-31`; training ends on `2019-12-01`. Re-run `make train` after changing the data to regenerate the numbers and artifact. Per-country results are in `data/output/model_metrics.json`; note that ARIMA is worse than the baseline for some individual countries (AU, FR, GB, US), while SARIMA beats the baseline for total revenue and for every country listed there. A model whose fit fails is reported with null metrics and an `error` field in `model_metrics.json` — it is never silently replaced by baseline numbers under its own name.

The API serves predictions from the SARIMA model: for a requested start date it refits SARIMA(2,1,2)(1,0,1,7) on the history available before that date and forecasts from there, because the saved artifact was trained on data ending 2019-12-01 and cannot answer an arbitrary start date directly. The response includes a `forecast_method` field (`sarima_refit` on success). Only if SARIMA fitting fails does the API fall back to a 7-day moving average, and then `forecast_method` is `baseline_fallback` — a flat fallback is never presented as a model forecast. A test asserts that a 5-day forecast is not a flat constant. A requested start date after the last available date is rejected with `date_out_of_range` (a forecast anchored at the end of the history must not be labelled with a later date); with no `date` parameter the API forecasts from the day after the last available date (`2020-01-01` for this sample), not from today.

## Operations and monitoring

`src/log.py` writes separate JSONL audit streams for ingestion, training, and prediction. The API adds a correlation ID to every response and prediction log. `src/monitor.py` calculates Wasserstein distance between a reference distribution and a current distribution, then returns the distance, configured threshold, drift decision, and either `continue` or `retrain`. The default threshold (250.0, in revenue units) is uncalibrated — calibrate it from the reference distribution's own scale before using drift decisions to trigger retraining.

## Known limitations

- The shipped data is a deterministic synthetic sample (one invoice per country per day), not real course invoices — Coursera provides no downloadable invoice files.
- The API refits SARIMA on every uncached request (~0.5 s for the sample series); there is no load test and no latency SLO beyond the measured local values.
- The drift-monitor threshold is an uncalibrated default (see above).
- The Docker image has not been build-verified in this environment (see "Docker"); the Dockerfile was fixed by inspection and its steps were verified locally.
- There is no CI workflow in the repository yet (see "Tests and quality gates").

The model artifact is treated as trusted deployment input. Before a real deployment, put the API behind TLS and authentication, restrict access to logs, add rate limiting, use a model registry with signed artifacts, and configure alerts around latency and drift.

## Tests and quality gates

```bash
make quality
make test
```

The suite contains 29 tests covering the API (including a test that SARIMA predictions are not flat, a test that unknown routes return 404 and not 500, and tests for out-of-range forecast dates), model behaviour (including a test that a failed fit is reported, not silently replaced), isolated logging, and ingestion of both the real Coursera and the legacy synthetic data schemas — including an end-to-end test that the shipped `data/input/` files ingest as the real Coursera schema with 26 files, 4,754 raw rows, 4,746 clean rows, and 8 dropped rows, reproducing the committed revenue series. Tests are self-contained: the API fixture trains a model into a temporary directory, so the suite passes on a fresh clone without running `make train` first, and tests write to temporary directories and never overwrite production `logs/` or `models/`. Run them with `python run_tests.py` — the last verified run passed 29/29 in a fresh virtualenv.

Note: the CI workflow file `.github/workflows/ci.yml` is **not in this repository yet**. The GitHub API integration used for publishing cannot create workflow files (it returns 403 for `.github/workflows/`). Add that file via the GitHub web UI to get automated ingestion, training, notebook validation, compilation, and tests on every push and pull request. The workflow must run `make train` before `make test` only if the tests depended on the artifact — they no longer do — but it still runs ingestion and training as a smoke check that the pipeline reproduces its metrics.

## Docker

```bash
docker build -t revenue-forecast-api .
docker run --rm -p 3000:80 --name revenue-forecast-api revenue-forecast-api
BASE_URL=http://127.0.0.1:3000 ./scripts/smoke_test.sh
```

The image runs as a non-root user, installs runtime dependencies only (`requirements.txt`; test and notebook tools live in `requirements-dev.txt`), and exposes a Docker healthcheck. A `.dockerignore` keeps `.git`, caches, local virtualenvs, generated artifacts, notebooks, and tests out of the build context (verified: the build context sent was 2.3 MB while the repository is 18 MB, mostly the gitignored 15 MB model artifact).

**Docker build status — not verified in this environment.** The Dockerfile was fixed by inspection (`.dockerignore`, runtime-only dependencies, documented build-time training), its syntax was validated with a Dockerfile parser (11 instructions), and every step it performs was verified locally: `pip install -r requirements.txt` in a fresh virtualenv, `run_ingest.py` + `run_train.py` reproducing the committed metrics, and `run_app.py` serving `/health` (ready) and `/predict` (`sarima_refit`, same values as the verified API run) with the Dockerfile's own healthcheck command returning 200. But `docker build` itself could **not** run here: the Docker daemon was installed (29.1.3) and started, yet the base image `python:3.11-slim` cannot be pulled in this VM — every container registry times out through the egress proxy (`registry-1.docker.io`, `ghcr.io`, `quay.io`, `mirror.gcr.io` all returned no response, while `pypi.org` answers normally). The build fails at step 1/11 (pulling the base image), before any project code runs. Verify with `docker build -t revenue-forecast-api .` on a machine with registry access before relying on the image for a demo. Note the image trains the SARIMA model at build time, so a successful build takes several minutes.

## Repository map

- `src/`: ingestion, modelling, API, logging, and monitoring modules.
- `tests/`: API, model, and logging tests.
- `nb/`: EDA and model-comparison notebooks.
- `data/input/`: monthly invoice JSON source files (synthetic sample, real Coursera schema).
- `data/output/`: generated deterministic data products and metrics.
- `models/`: trained artifact (generated locally by `make train`, gitignored).
- `requirements.txt`: runtime dependencies (installed in the Docker image).
- `requirements-dev.txt`: development, test, and notebook dependencies.
- `.dockerignore`: build-context exclusions for the Docker image.
- `.github/workflows/ci.yml`: automated quality gate (**not in the repository yet** — see "Tests and quality gates").
- `.env.example`: documented runtime configuration.
- `SECURITY.md`: deployment security expectations.
