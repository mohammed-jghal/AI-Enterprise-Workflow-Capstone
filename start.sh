#!/bin/sh
set -e
python run_ingest.py
python run_train.py
python run_app.py
