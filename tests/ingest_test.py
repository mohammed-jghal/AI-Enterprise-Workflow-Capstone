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


def test_shipped_data_is_real_coursera_schema(tmp_path):
    """End-to-end guard for the shipped sample: the files in data/input
    must be in the real Coursera schema (not the legacy synthetic one),
    with the same row counts the pipeline was measured on."""
    repo_root = Path(__file__).parents[1]
    out = tmp_path / "output"
    result = ingest(str(repo_root / "data" / "input"), str(out))
    assert result["schema"] == "coursera"
    assert result["files_read"] == 26
    assert result["raw_rows"] == 4754
    assert result["clean_rows"] == 4746
    assert result["rows_dropped"] == 8
    total = pd.read_csv(out / "4 revenue_total.csv")
    assert len(total) == 791  # 2017-11-01 .. 2019-12-31, daily
    assert list(total["invoice_date"])[0] == "2017-11-01"
    assert list(total["invoice_date"])[-1] == "2019-12-31"
    # The real-schema ingest must reproduce the committed revenue series.
    committed = pd.read_csv(repo_root / "data" / "output" / "4 revenue_total.csv")
    assert total["revenue"].tolist() == committed["revenue"].tolist()


def test_detect_schema_majority_vote():
    coursera = [{"country": "France", "price": 1.0, "year": 2018, "month": 1, "day": 2}]
    synthetic = [{"invoice_date": "2018-01-02", "country": "FR", "amount": 1.0}]
    assert detect_schema(coursera * 3 + synthetic) == "coursera"
    assert detect_schema(synthetic * 3 + coursera) == "synthetic"
    assert detect_schema([{"foo": 1}]) == "unknown"


def test_real_schema_bad_rows_are_dropped_and_counted(tmp_path):
    records = [
        {"country": "France", "customer_id": 1.0, "invoice": "1", "price": 5.0,
         "stream_id": "1", "times_viewed": 2, "year": "2018", "month": "1", "day": "2"},
        {"country": None, "customer_id": None, "invoice": "bad-1", "price": None,
         "stream_id": None, "times_viewed": None, "year": None, "month": None, "day": None},
    ]
    input_dir = _write_invoices(tmp_path, "invoices-2018-01.json", records)
    result = ingest(str(input_dir), str(tmp_path / "output"))
    assert result["raw_rows"] == 2
    assert result["clean_rows"] == 1
    assert result["rows_dropped"] == 1
