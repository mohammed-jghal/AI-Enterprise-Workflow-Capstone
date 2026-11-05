"""JSONL logging with injectable directories for test isolation."""
from pathlib import Path
from datetime import datetime, timezone
import json

def write_log(kind, event, log_dir='logs'):
    if kind not in {'ingest','train','predict'}: raise ValueError('kind must be ingest, train, or predict')
    path=Path(log_dir); path.mkdir(parents=True, exist_ok=True)
    entry={'timestamp':datetime.now(timezone.utc).isoformat(),'level':'INFO','event':event}
    with (path/f'{kind}.jsonl').open('a') as f: f.write(json.dumps(entry, default=str)+'\n')
    return entry

def read_log(kind, log_dir='logs'):
    file=Path(log_dir)/f'{kind}.jsonl'
    if not file.exists(): return []
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
