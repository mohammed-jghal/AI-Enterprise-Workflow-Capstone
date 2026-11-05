"""Run ingestion. Resolves paths from the repository root, so this script
works from any working directory (previously it used cwd-relative paths
and failed when run from outside the repo root)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ingest import ingest
from src.log import write_log

result = ingest(str(ROOT / "data" / "input"), str(ROOT / "data" / "output"))
write_log("ingest", result, str(ROOT / "logs"))
print(result)
