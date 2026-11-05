#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:80}"
curl --fail --silent "$BASE_URL/health" | python3 -m json.tool
curl --fail --silent -X POST "$BASE_URL/predict?date=2018-11-20&duration=7" | python3 -m json.tool
