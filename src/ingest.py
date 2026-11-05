"""Ingest monthly invoice JSON files and produce deterministic feature tables.

Supports two input schemas:

1. Real Coursera course schema (IBM AI Enterprise Workflow, as published
   in the reference repository) — also the schema of the synthetic sample
   shipped in ``data/input/``:
   ``country`` (full name, e.g. ``"United Kingdom"``), ``price`` (numeric),
   ``times_viewed``, ``year`` / ``month`` / ``day``, ``invoice``,
   ``customer_id``, ``stream_id``.
   Revenue for a record is ``price``; the invoice date is built from
   ``year``/``month``/``day``; full country names are mapped to ISO codes.

2. Legacy synthetic schema (kept for backward compatibility):
   ``invoice_date`` (ISO date), ``country`` (two-letter code, e.g. ``US``),
   ``amount`` (numeric revenue).

Both schemas produce the same five deterministic outputs, so downstream
code (model, API, notebooks) is unchanged whichever input is used.
"""
from pathlib import Path
import json
import pandas as pd

OUTPUTS = ['0 data.csv', '1 data_cleaned.csv', '2 data_engineered.csv',
           '3 revenue_country.csv', '4 revenue_total.csv']

# Full country name (lowercase, stripped) -> ISO 3166-1 alpha-2 code.
COUNTRY_MAP = {
    'united kingdom': 'GB', 'uk': 'GB', 'great britain': 'GB', 'england': 'GB',
    'united states': 'US', 'usa': 'US', 'us': 'US', 'united states of america': 'US',
    'france': 'FR', 'germany': 'DE', 'deutschland': 'DE', 'canada': 'CA',
    'australia': 'AU', 'netherlands': 'NL', 'holland': 'NL', 'spain': 'ES',
    'italy': 'IT', 'ireland': 'IE', 'belgium': 'BE', 'switzerland': 'CH',
    'austria': 'AT', 'sweden': 'SE', 'norway': 'NO', 'denmark': 'DK',
    'finland': 'FI', 'portugal': 'PT', 'greece': 'GR', 'poland': 'PL',
    'japan': 'JP', 'south korea': 'KR', 'korea': 'KR', 'india': 'IN',
    'brazil': 'BR', 'mexico': 'MX', 'new zealand': 'NZ', 'south africa': 'ZA',
    'china': 'CN', 'russia': 'RU', 'russian federation': 'RU',
    'turkey': 'TR', 'israel': 'IL', 'egypt': 'EG', 'morocco': 'MA',
    'argentina': 'AR', 'chile': 'CL', 'colombia': 'CO', 'peru': 'PE',
    'czech republic': 'CZ', 'czechia': 'CZ', 'hungary': 'HU', 'romania': 'RO',
    'ukraine': 'UA', 'iceland': 'IS', 'luxembourg': 'LU', 'malta': 'MT',
}


def map_country(raw):
    """Map a country value (full name or two-letter code) to an ISO code."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in COUNTRY_MAP:
        return COUNTRY_MAP[lowered]
    if len(text) == 2 and text.isalpha():
        return text.upper()
    return text.upper()


def normalize_record(record):
    """Return a dict with normalized invoice_date / country / amount.

    Keeps the original fields as well, so ``0 data.csv`` preserves the
    source record. Returns None values for fields that cannot be derived;
    those rows are dropped in the cleaning step and counted.
    """
    out = dict(record) if isinstance(record, dict) else {}
    # --- invoice date -------------------------------------------------
    date = record.get('invoice_date') if isinstance(record, dict) else None
    if date is None and isinstance(record, dict):
        year = record.get('year')
        month = record.get('month')
        day = record.get('day')
        if year is not None and month is not None and day is not None:
            try:
                date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            except (TypeError, ValueError):
                date = None
    out['invoice_date'] = date
    # --- amount / revenue --------------------------------------------
    amount = None
    if isinstance(record, dict):
        for key in ('amount', 'price', 'total_price'):
            if record.get(key) is not None:
                amount = record.get(key)
                break
    out['amount'] = amount
    # --- country ------------------------------------------------------
    country = record.get('country') if isinstance(record, dict) else None
    out['country'] = map_country(country)
    # --- invoice id (reference schema uses ``invoice``) --------------
    if 'invoice_id' not in out and isinstance(record, dict):
        out['invoice_id'] = record.get('invoice_id', record.get('invoice'))
    return out


def detect_schema(records):
    """Return 'coursera' if records look like the real course schema.

    Majority vote over the records, not just the first match: a file
    with a stray header row or a bad row first must not mislabel the
    whole file. Returns 'unknown' when neither schema matches.
    """
    coursera = synthetic = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if 'price' in rec and ('year' in rec or 'times_viewed' in rec):
            coursera += 1
        elif 'invoice_date' in rec and 'amount' in rec:
            synthetic += 1
    if coursera and coursera >= synthetic:
        return 'coursera'
    if synthetic:
        return 'synthetic'
    return 'unknown'


def ingest(input_dir='data/input', output_dir='data/output'):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob('invoices-*.json'))
    records = []
    for f in files:
        payload = json.loads(f.read_text())
        if isinstance(payload, dict):
            payload = payload.get('invoices', payload.get('data', []))
        records.extend(payload)
    schema = detect_schema(records)
    normalized = [normalize_record(r) for r in records]
    raw = pd.DataFrame(normalized)
    for col in ['invoice_date', 'country', 'amount']:
        if col not in raw:
            raw[col] = None
    raw.to_csv(output_dir / OUTPUTS[0], index=False)
    cleaned = raw.copy()
    cleaned['invoice_date'] = pd.to_datetime(cleaned['invoice_date'], errors='coerce')
    cleaned['amount'] = pd.to_numeric(cleaned['amount'], errors='coerce')
    cleaned['country'] = cleaned['country'].astype('string').str.strip().str.upper()
    cleaned = cleaned.dropna(subset=['invoice_date', 'country', 'amount'])
    cleaned = cleaned[cleaned['amount'] >= 0].copy()
    cleaned['invoice_date'] = cleaned['invoice_date'].dt.strftime('%Y-%m-%d')
    cleaned.to_csv(output_dir / OUTPUTS[1], index=False)
    engineered = cleaned.copy()
    dt = pd.to_datetime(engineered['invoice_date'])
    engineered['year'] = dt.dt.year
    engineered['month'] = dt.dt.month
    engineered['day'] = dt.dt.day
    engineered['day_of_week'] = dt.dt.dayofweek
    engineered['country_code'] = engineered['country'].astype('category').cat.codes
    engineered.to_csv(output_dir / OUTPUTS[2], index=False)
    country = cleaned.groupby(['invoice_date', 'country'], as_index=False)['amount'].sum().rename(columns={'amount': 'revenue'})
    country.to_csv(output_dir / OUTPUTS[3], index=False)
    total = cleaned.groupby('invoice_date', as_index=False)['amount'].sum().rename(columns={'amount': 'revenue'})
    total['invoice_date'] = pd.to_datetime(total['invoice_date'])
    total = total.set_index('invoice_date').asfreq('D', fill_value=0).reset_index()
    total.to_csv(output_dir / OUTPUTS[4], index=False)
    return {'files_read': len(files), 'raw_rows': len(raw), 'clean_rows': len(cleaned),
            'rows_dropped': len(raw) - len(cleaned), 'schema': schema,
            'output_dir': str(output_dir)}


if __name__ == '__main__':
    print(ingest())
