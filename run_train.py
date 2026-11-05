from pathlib import Path
import json
import pandas as pd
from src.model import train_and_save, evaluate_series
from src.log import write_log
p=Path('data/output/4 revenue_total.csv')
if not p.exists():
    from src.ingest import ingest; ingest()
df=pd.read_csv(p, parse_dates=['invoice_date'])
payload=train_and_save(df['revenue'])
total_metrics=evaluate_series(df.set_index('invoice_date')['revenue'])
country_df=pd.read_csv('data/output/3 revenue_country.csv', parse_dates=['invoice_date'])
country_metrics={country:evaluate_series(group.set_index('invoice_date')['revenue']) for country, group in country_df.groupby('country')}
metrics={'total':total_metrics,'by_country':country_metrics}
write_log('train', {'model_type':'SARIMA','model_version':payload['model_version'],'artifact':'models/revenue_model.pkl','metrics':metrics})
Path('data/output/model_metrics.json').write_text(json.dumps(metrics, indent=2))
print({'artifact':'models/revenue_model.pkl','metrics':metrics})
