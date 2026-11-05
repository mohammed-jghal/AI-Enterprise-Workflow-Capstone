PYTHON ?= python3

.PHONY: install ingest train test quality run docker

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

ingest:
	$(PYTHON) run_ingest.py

train: ingest
	$(PYTHON) run_train.py

test:
	$(PYTHON) run_tests.py

quality:
	$(PYTHON) -m compileall -q src tests run_*.py
	$(PYTHON) -m json.tool nb/analysis.ipynb >/dev/null
	$(PYTHON) -m json.tool nb/results.ipynb >/dev/null

run: train
	$(PYTHON) run_app.py

docker:
	docker build -t revenue-forecast-api .
