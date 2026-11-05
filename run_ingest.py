from src.ingest import ingest
from src.log import write_log
result=ingest()
write_log("ingest", result)
print(result)
