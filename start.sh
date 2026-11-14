source .venv/bin/activate
# start prefect
uv run prefect server start &
# start mlflow (port 5001)
uv run mlflow server --port 5001 --backend-store-uri sqlite:///mlflow.db &

uv run prefect worker start --pool mlops-pipeline &

wait