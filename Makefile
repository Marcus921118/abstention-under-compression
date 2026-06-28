# Windows note: run via `make` if available, else use the equivalent commands
# directly. PY points at the project venv interpreter.
PY ?= python

.PHONY: setup setup-ml test lint typecheck selftest clean

setup:                ## install core (no ML) deps + package in editable mode
	$(PY) -m pip install -e ".[dev]"

setup-ml:             ## install heavy ML stack (use a Python 3.10-3.12 env!)
	$(PY) -m pip install -e ".[ml]"

test:                 ## run unit tests (no model required)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

selftest:             ## exercise the metric stack on synthetic data
	$(PY) -m aq.cli selftest

clean:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
