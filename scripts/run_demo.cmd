@echo off

set "SIGNALRANK_SERVICE_TOKEN=signalrank-local-dev"
set "SIGNALRANK_CONFIG=configs\benchmark.yaml"

start "SignalRank API" cmd /k "uv run uvicorn signalrank.api.main:app --reload"
start "SignalRank UI" cmd /k "uv run streamlit run src\signalrank\ui\app.py"