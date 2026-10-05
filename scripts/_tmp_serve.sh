#!/usr/bin/env bash
cd /home/su127/FYP/domain-bounded-cslr
export PYTHONPATH=src
exec ./venv/bin/python3 -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000