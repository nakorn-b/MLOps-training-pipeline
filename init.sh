#!/bin/bash
# Harness verification: run before starting work and before claiming a feature done.
set -e
cd "$(dirname "$0")"

echo "=== Harness Initialization ==="

echo "=== uv sync ==="
uv sync

echo "=== import check ==="
MLFLOW_DISABLE_AGENT_HINT=1 uv run python -c "import main, model_factory, pipeline.flow"

# End-to-end train flow. Uses a temporary Prefect server and a throwaway MLflow
# database, so it needs neither start.sh nor touches mlflow.db.
if [ -d data/cifar10-mini ]; then
  echo "=== end-to-end train flow (configs/example.yaml) ==="
  tmp=$(mktemp -d)
  PREFECT_API_URL= \
  PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=true \
  PREFECT_LOGGING_LEVEL=WARNING \
  MLFLOW_DISABLE_AGENT_HINT=1 \
  MLFLOW_TRACKING_URI="sqlite:///$tmp/mlflow.db" \
    uv run main.py --config configs/example.yaml

  echo "=== MLflow run was logged ==="
  MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_TRACKING_URI="sqlite:///$tmp/mlflow.db" uv run python -c "
import mlflow
runs = mlflow.search_runs(search_all_experiments=True)
assert len(runs) == 1, f'expected 1 run, found {len(runs)}'
row = runs.iloc[0]
assert row['status'] == 'FINISHED', row['status']
print(row.filter(regex='^(metrics|params)').to_string())

# A metric logged from inside Trainer.train() must land in the same run as the
# final metrics (the flow opens one run before training).
import pandas as pd
assert pd.notna(row.get('metrics.train_accuracy')), 'metric logged in Trainer.train() is not in the run'
print('train-time metric in the same run OK')

# The logged model must reload in the generic pyfunc flavor and predict.
models = mlflow.search_logged_models(experiment_ids=[row['experiment_id']], output_format='list')
assert len(models) == 1, f'expected 1 logged model, found {len(models)}'
uri = models[0].model_uri
example = mlflow.models.Model.load(uri).load_input_example(mlflow.artifacts.download_artifacts(uri))
pred = mlflow.pyfunc.load_model(uri).predict(example)
print(f'pyfunc reload OK: {uri} -> {list(pred)}')

# The trainer must be able to start from that logged model.
from pipeline.config import Config
from trainers.logistic_regression import LogisticRegressionTrainer
t = LogisticRegressionTrainer(Config(data={'uri': ''}, model={'name': 'logistic_regression'}))
t.load(uri)
assert list(t.model.predict(example)) == list(pred)
print('trainer.load(model_uri) OK')

# evaluate must log macro OvR ROC AUC and the ROC curve plot to the same run.
assert pd.notna(row.get('metrics.test_roc_auc')), 'test_roc_auc missing from the run'
artifacts = [a.path for a in mlflow.artifacts.list_artifacts(run_id=row['run_id'])]
assert 'roc_curve.png' in artifacts, f'roc_curve.png not in run artifacts: {artifacts}'
print(f\"ROC OK: test_roc_auc={row['metrics.test_roc_auc']:.4f}, roc_curve.png logged\")
"
  rm -rf "$tmp"
else
  echo "!!! SKIPPED end-to-end flow: data/cifar10-mini is missing (data/ is gitignored)."
  echo "!!! Only the import check ran. Do not mark a feature done on this result."
fi

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read feature_list.json to see current feature state"
echo "2. Pick ONE unfinished feature to work on"
echo "3. Implement only that feature"
echo "4. Re-run ./init.sh before claiming done"
