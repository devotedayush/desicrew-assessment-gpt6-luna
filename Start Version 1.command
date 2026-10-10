#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ -x .venv-v1-py312/bin/python ]]; then
  exec .venv-v1-py312/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
fi
exec python3.12 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
