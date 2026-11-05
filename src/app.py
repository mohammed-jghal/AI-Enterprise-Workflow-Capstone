"""Production Flask API for revenue forecasts.

The application is deliberately small, but follows production conventions:
configuration through environment variables, deterministic validation, a health
endpoint, consistent error envelopes, cached read-only datasets, and request
correlation IDs in logs.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
import uuid
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, g, jsonify, request
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .log import read_log, write_log

BASE = Path(os.getenv("APP_HOME", Path(__file__).resolve().parents[1])).resolve()
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "models/revenue_model.pkl"))
COUNTRY_PATH = Path(os.getenv("COUNTRY_DATA_PATH", BASE / "data/output/3 revenue_country.csv"))
TOTAL_PATH = Path(os.getenv("TOTAL_DATA_PATH", BASE / "data/output/4 revenue_total.csv"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE / "logs"))
MAX_HORIZON = int(os.getenv("MAX_FORECAST_DAYS", "365"))


def create_app(test_config: dict | None = None) -> Flask:
    application = Flask(__name__)
    application.config.update(
        JSON_SORT_KEYS=True,
        MAX_FORECAST_DAYS=MAX_HORIZON,
        MODEL_PATH=MODEL_PATH,
        COUNTRY_PATH=COUNTRY_PATH,
        TOTAL_PATH=TOTAL_PATH,
        LOG_DIR=LOG_DIR,
    )
    if test_config:
        application.config.update(test_config)

    @application.before_request
    def attach_request_id():
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.started_at = time.perf_counter()

    @application.after_request
    def add_response_headers(response):
        response.headers["X-Request-ID"] = g.get("request_id", "unknown")
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.errorhandler(Exception)
    def handle_unexpected_error(exc):
        application.logger.exception("Unhandled request error", exc_info=exc)
        return _error("internal_error", "An unexpected server error occurred", 500)

    @application.get("/health")
    def health():
        model_path = Path(application.config["MODEL_PATH"])
        total_path = Path(application.config["TOTAL_PATH"])
        ready = model_path.exists() and total_path.exists()
        result = {
            "status": "ready" if ready else "degraded",
            "service": "revenue-forecast-api",
            "version": _model_version(model_path),
            "dependencies": {"model_artifact": model_path.exists(), "total_dataset": total_path.exists()},
        }
        return jsonify(result), 200 if ready else 503

    @application.get("/version")
    def version():
        return jsonify({"service": "revenue-forecast-api", "model_version": _model_version(Path(application.config["MODEL_PATH"]))})

    @application.post("/predict")
    def predict():
        started = time.perf_counter()
        try:
            start = _parse_date(request.args.get("date"))
            duration = _parse_duration(request.args.get("duration", "30"), application.config["MAX_FORECAST_DAYS"])
            country = _parse_country(request.args.get("country"))
            data = _load_payload(country, application.config)
            history = data[data["invoice_date"] < start]
            if history.empty:
                raise ApiError("date_out_of_range", "date is before available history", 400)
            predictions, forecast_method = _forecast(
                history, duration, Path(application.config["MODEL_PATH"])
            )
            end = start + pd.Timedelta(days=duration - 1)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            payload = {
                "request_id": g.request_id,
                "model_version": _model_version(Path(application.config["MODEL_PATH"])),
                "country": country or "TOTAL",
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "duration_days": duration,
                "daily_predictions": predictions,
                "total_prediction": round(sum(predictions), 2),
                "latency_ms": latency_ms,
                "forecast_method": forecast_method,
            }
            write_log("predict", payload, application.config["LOG_DIR"])
            return jsonify(payload)
        except ApiError as exc:
            return _error(exc.code, exc.message, exc.status)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            return _error("invalid_request", str(exc), 400)

    @application.post("/logs")
    def logs():
        kind = request.args.get("type", "predict")
        if kind not in {"ingest", "train", "predict"}:
            return _error("invalid_log_type", "type must be ingest, train, or predict", 400)
        return jsonify({"type": kind, "logs": read_log(kind, application.config["LOG_DIR"])})

    return application


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int):
        self.code, self.message, self.status = code, message, status


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message, "request_id": getattr(g, "request_id", None)}}), status


def _parse_date(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.Timestamp.today().normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ApiError("invalid_date", "date must be a valid ISO date such as 2018-11-20", 400)
    return parsed.normalize()


def _parse_duration(value: str, maximum: int) -> int:
    try:
        duration = int(value)
    except (TypeError, ValueError):
        raise ApiError("invalid_duration", "duration must be an integer", 400)
    if not 1 <= duration <= maximum:
        raise ApiError("invalid_duration", f"duration must be between 1 and {maximum}", 400)
    return duration


def _parse_country(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    country = value.strip().upper()
    if len(country) != 2 or not country.isalpha():
        raise ApiError("invalid_country", "country must be a two-letter ISO-style code", 400)
    return country


@lru_cache(maxsize=8)
def _read_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    frame = pd.read_csv(path)
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    return frame.sort_values("invoice_date")


def _load_payload(country: str | None, config: dict) -> pd.DataFrame:
    path = Path(config["COUNTRY_PATH"] if country else config["TOTAL_PATH"])
    frame = _read_csv(str(path)).copy()
    if country:
        frame = frame[frame["country"].str.upper() == country]
        if frame.empty:
            raise ApiError("unknown_country", f"No data found for country {country}", 400)
        frame = frame.groupby("invoice_date", as_index=False)["revenue"].sum()
    return frame


_FORECAST_CACHE: dict = {}


def _saved_model_spec(model_path: Path) -> dict:
    """Read the saved model payload's spec (kind, orders, version).

    Returns the SARIMA order / seasonal order the model was trained with,
    so the API forecasts with the same model family as the trained
    artifact instead of silently substituting a different method.
    Defaults match ``src/model.py`` when no artifact is present.
    """
    spec = {"kind": "sarima", "order": (2, 1, 2), "seasonal_order": (1, 0, 1, 7)}
    try:
        with model_path.open("rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, dict):
            spec["kind"] = str(payload.get("kind", "sarima"))
            if payload.get("order"):
                spec["order"] = tuple(payload["order"])
            if payload.get("seasonal_order"):
                spec["seasonal_order"] = tuple(payload["seasonal_order"])
    except Exception:
        pass
    return spec


def _forecast(frame: pd.DataFrame, horizon: int, model_path: Path | None = None) -> tuple[list[float], str]:
    """Forecast with the trained SARIMA model.

    The saved artifact was trained on data ending 2019-12-01, so its
    fitted values cannot answer an arbitrary requested start date. The
    correct production behaviour is therefore to refit the same SARIMA
    specification on the history available before the requested start
    date and forecast from there. The saved artifact supplies the model
    family, orders, and version; the refit supplies the date anchor.

    Returns ``(predictions, method)`` where method is ``"sarima_refit"``
    on success or ``"baseline_fallback"`` when SARIMA fitting fails
    (the fallback is logged by the caller through the payload).
    A flat 7-day mean is never returned silently as a model forecast.
    """
    values = frame["revenue"].astype(float).to_numpy()
    if len(values) < 7:
        raise ApiError("insufficient_history", "at least seven historical days are required", 400)
    cache_key = (
        len(values),
        str(frame["invoice_date"].iloc[-1].date()),
        horizon,
        hashlib.sha256(np.asarray(values, dtype=float).tobytes()).hexdigest(),
    )
    if cache_key in _FORECAST_CACHE:
        return _FORECAST_CACHE[cache_key]
    spec = _saved_model_spec(model_path) if model_path is not None else {
        "kind": "sarima", "order": (2, 1, 2), "seasonal_order": (1, 0, 1, 7),
    }
    predictions: list[float] | None = None
    method = "baseline_fallback"
    if spec["kind"] == "sarima" and len(values) >= 60:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = SARIMAX(
                    values,
                    order=spec["order"],
                    seasonal_order=spec["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                raw = np.asarray(fitted.forecast(horizon), dtype=float)
            predictions = [round(float(v), 2) for v in raw]
            method = "sarima_refit"
        except Exception:
            predictions = None
    if predictions is None:
        window = min(7, len(values))
        value = round(float(values[-window:].mean()), 2)
        predictions = [value] * horizon
        method = "baseline_fallback"
    result = (predictions, method)
    _FORECAST_CACHE[cache_key] = result
    return result


def _model_version(path: Path) -> str:
    if not path.exists():
        return "unavailable"
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        return str(payload.get("model_version", "unknown"))
    except Exception:
        return "invalid"


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "80")))
