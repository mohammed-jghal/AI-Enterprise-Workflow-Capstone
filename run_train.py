"""Train the SARIMA model and write metrics. Paths resolve from the
repository root, so this script works from any working directory."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.model import train_and_save, evaluate_series
from src.log import write_log

p = ROOT / "data" / "output" / "4 revenue_total.csv"
if not p.exists():
    from src.ingest import ingest
    ingest(str(ROOT / "data" / "input"), str(ROOT / "data" / "output"))
df = pd.read_csv(p, parse_dates=["invoice_date"])
payload = train_and_save(df["revenue"], str(ROOT / "models" / "revenue_model.pkl"))
total_metrics = evaluate_series(df.set_index("invoice_date")["revenue"])
country_df = pd.read_csv(ROOT / "data" / "output" / "3 revenue_country.csv",
                          parse_dates=["invoice_date"])
country_metrics = {country: evaluate_series(group.set_index("invoice_date")["revenue"])
                    for country, group in country_df.groupby("country")}
metrics = {"total": total_metrics, "by_country": country_metrics}
write_log("train", {"model_type": "SARIMA", "model_version": payload["model_version"],
                    "artifact": "models/revenue_model.pkl", "metrics": metrics},
          str(ROOT / "logs"))
(ROOT / "data" / "output" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
failed = [row["model"] for section in (total_metrics, *country_metrics.values())
          for row in section if row.get("error")]
if failed:
    print(f"WARNING: model evaluation failed for: {sorted(set(failed))} "
          f"(see 'error' in data/output/model_metrics.json)")
print({"artifact": "models/revenue_model.pkl", "metrics": metrics})
