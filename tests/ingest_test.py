import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import pandas as pd

from src.ingest import ingest, normalize_record, detect_schema


def _write_invoices(tmp_path, name, records):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / name).write_text(json.dumps(records))
    return input_dir


def test_real_coursera_schema(tmp_path):
    records = [
        {"country": "United Kingdom", "customer_id": 13085.0, "invoice": "489434",
         "price": 6.95, "stream_id": "85048", "times_viewed": 12,
         "year": "2017", "month": "11", "day": "28"},
        {"country": "United Kingdom", "customer_id": None, "invoice": "489597",
         "price": 8.65, "stream_id": "22130", "times_viewed": 1,
         "year": "2017", "month": "11", "day": "28"},
        {"country": "United States", "customer_id": 14001.0, "invoice": "489598",
         "price": 10.00, "stream_id": "22131", "times_viewed": 3,
         "year": "2017", "month": "11", "day": "29"},
    ]
    assert detect_schema(records) == "coursera"
    input_dir = _write_invoices(tmp_path, "invoices-2017-11.json", records)
    out = tmp_path / "output"
    result = ingest(str(input_dir), str(out))
    assert result["schema"] == "coursera"
    assert result["files_read"] == 1
    assert result["clean_rows"] == 3
    assert result["rows_dropped"] == 0
    total = pd.read_csv(out / "4 revenue_total.csv")
    assert list(total["invoice_date"]) == ["2017-11-28", "2017-11-29"]
    assert abs(total["revenue"].iloc[0] - 15.60) < 1e-6
    assert abs(total["revenue"].iloc[1] - 10.00) < 1e-6
    country = pd.read_csv(out / "3 revenue_country.csv")
    assert set(country["country"]) == {"GB", "US"}
    assert abs(country[country["country"] == "GB"]["revenue"].sum() - 15.60) < 1e-6


def test_synthetic_schema_still_works(tmp_path):
    records = [
        {"invoice_id": "201711-00000", "invoice_date": "2017-11-01",
         "country": "US", "amount": 524.92, "currency": "USD"},
        {"invoice_id": "201711-00001", "invoice_date": "2017-11-01",
         "country": "GB", "amount": 277.09, "currency": "USD"},
    ]
    assert detect_schema(records) == "synthetic"
    input_dir = _write_invoices(tmp_path, "invoices-2017-11.json", records)
    out = tmp_path / "output"
    result = ingest(str(input_dir), str(out))
    assert result["schema"] == "synthetic"
    assert result["clean_rows"] == 2
    total = pd.read_csv(out / "4 revenue_total.csv")
    assert abs(total["revenue"].iloc[0] - 802.01) < 1e-6


def test_normalize_record_maps_full_country_names():
    rec = normalize_record({"country": "United Kingdom", "price": 5.0,
                            "year": 2017, "month": 11, "day": 28})
    assert rec["country"] == "GB"
    assert rec["invoice_date"] == "2017-11-28"
    assert rec["amount"] == 5.0
