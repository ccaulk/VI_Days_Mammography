PY := .venv/bin/python

.PHONY: setup weights data test run

setup:
	uv venv --python 3.12 .venv
	uv sync --all-groups

weights: ; $(PY) scripts/fetch_weights.py
data:    ; $(PY) scripts/fetch_data.py
test:    ; $(PY) -m pytest -q
run:     ; .venv/bin/streamlit run asymview/app.py
