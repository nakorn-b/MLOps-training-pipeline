# training-pipeline

Automated train/retrain pipeline orchestrated with Prefect. A dev supplies one
experiment YAML and the run goes end to end on its own: preprocess, train,
evaluate, register in MLflow. Design sketch: `plan.png`.

Status: **skeleton**. Signatures exist, bodies raise `NotImplementedError`.
Task and `Trainer` method arguments are deliberately undecided; do not invent
them without asking.

## Agent harness

### Startup Workflow

Before writing code:

1. Read `feature_list.json` and `progress.md` (and `session-handoff.md` if recent).
2. Run `./init.sh`. If it fails, fix that before adding scope.
3. **One feature at a time**: pick one unfinished feature from `feature_list.json`; respect its
   `dependencies`. Stay in scope: do not change files unrelated to that feature.

`./init.sh` runs `uv sync`, an import check, and the train flow end to end on
`data/cifar10-mini` with a temporary Prefect server and a throwaway MLflow
database. It needs no running servers and never writes to `mlflow.db`.

### Definition of Done

A feature is done only when all of these are true:

- its "Done when" checks in `feature_list.json` pass;
- `./init.sh` is green, including the end-to-end step (not skipped);
- the command and result are recorded in the feature's `evidence` field.

### End of Session

Before ending: update `feature_list.json` and `progress.md` (and
`session-handoff.md` for larger sessions), and note any blocker that leaves
`./init.sh` red.

## Commands

```bash
uv run main.py --config configs/example.yaml                              # train
uv run main.py --config configs/example.yaml --retrain                    # retrain
uv run main.py --config configs/example.yaml --retrain --dataset-uri <uri>
```

Use `uv` for everything (`uv add <pkg>`, `uv run ...`). Python 3.13.

## Layout

```
main.py              CLI entry, picks the train or retrain flow
model_factory.py     TRAINERS registry (name -> Trainer class) + get_trainer
trainers/base.py     Trainer(ABC): load, preprocess, train, predict
trainers/dino.py     DinoTrainer  ("dinov3")
trainers/yolo.py     YoloTrainer  ("yolo")
pipeline/config.py   pydantic Config (data, model, params sections)
pipeline/tasks.py    Prefect tasks: load_config, preprocess, train, evaluate, register
pipeline/flow.py     Prefect flows: train, retrain
configs/             one YAML per experiment
```

## Design rules

- **Pipeline code is model-agnostic.** Tasks only talk to the `Trainer`
  interface. Anything model-specific goes inside a trainer, never in `pipeline/`.
- **Adding a model** = one `Trainer` subclass in `trainers/` + one entry in
  `TRAINERS`. Nothing else should need to change.
- **`TRAINERS` maps names to classes, not instances.** `get_trainer` builds only
  the requested one; an unknown name raises an error listing the valid names.
- **One YAML is the only experiment input.** Infrastructure settings stay out of
  it: `register` reads the MLflow tracking URI from `MLFLOW_TRACKING_URI`
  (fallback `sqlite:///mlflow.db`) and the experiment name from
  `EXPERIMENT_NAME` (fallback `MLOps-pipeline`). Both are set in `.env`, which
  `pipeline/tasks.py` loads on import.
- **Config is typed.** `load_config` returns a pydantic `Config`, so bad YAML
  fails before training starts. The data location is `data.uri`.
- **Hyperparameters live under `params:`** in the YAML as a free-form dict,
  because the keys are model-specific. The trainer passes them to its model and
  validates them. `register` logs the whole YAML to MLflow as flattened params
  (`data.uri`, `model.name`, `params.<key>`) plus the metrics.
- **Data arrives unsplit**: `data.uri` is a folder with one subfolder per class.
  The `preprocess` task does two things, in order:
  1. Split: assign every image to train / val / test (80/10/10, stratified by
     class, fixed seed) in a manifest DataFrame with columns `path`, `label`,
     `split`. No files are moved or copied. This part is generic pipeline code.
  2. Model-specific preparation: pass the manifest to `trainer.preprocess()`.
- **Task scope is image classification** for now (YOLO `-cls`, DINOv3 backbone
  + classifier head).

## Trainer methods

| Method | Meaning |
|---|---|
| `load(model_uri=None)` | Build the model. `None`: fresh/pretrained weights; otherwise start from that MLflow model URI (the flow resolves it, e.g. the champion). Not data loading. |
| `preprocess(manifest_df)` | Prepare the split dataset for this model (resize, normalise, convert format). |
| `train()` | Train. |
| `predict()` | Predict class labels for the test split. |
| `log_model()` | Log the trained model to the active MLflow run in this model's flavor; return its model URI. Called by `register`. |

Trainers also expose `y_test` (true test labels), which the `evaluate` task reads.

`evaluate()` is planned future work. Until then the `evaluate` task is generic
pipeline code: it scores the test split and returns accuracy and
macro-averaged precision, recall and F1.

## Flows

Both flows run the same tasks: `load_config -> preprocess -> train -> evaluate -> register`.

| | `train(config_path)` | `retrain(config_path, dataset_uri=None)` |
|---|---|---|
| Starting weights | pretrained | current MLflow champion; error if none exists |
| Data | `data.uri` from the YAML | `dataset_uri` if given, else the YAML |
| Champion comparison | against the champion's logged metric | champion and new model both scored on the new test split |

`register` always logs the run to MLflow but promotes the model only when it
beats the champion.

Automated retraining keeps the YAML fixed and passes new data through
`dataset_uri`. Triggers (schedule, new data, drift) are not built yet; they
will be Prefect deployments.

## Intended stack (not installed yet)

`mlflow`, `ultralytics`, `torch`. Add each with `uv add` when its step is
implemented. `pydantic` is already available through Prefect; add it explicitly
when `Config` gets real use.
