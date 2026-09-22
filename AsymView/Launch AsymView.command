#!/bin/zsh
cd "${0:A:h}"
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
