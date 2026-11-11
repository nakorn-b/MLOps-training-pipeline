# training-pipeline

Automated train/retrain pipeline orchestrated with Prefect. A dev supplies one
experiment YAML and the run goes end to end on its own: preprocess, train,
evaluate, register in MLflow. Design sketch: `plan.png`.

Status: **skeleton**. Signatures exist, bodies raise `NotImplementedError`.
Task and `Trainer` method arguments are deliberately undecided; do not invent
them without asking.

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
pipeline/config.py   pydantic Config (data, model, train sections)
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
  it: the MLflow tracking URI comes from `MLFLOW_TRACKING_URI` (fallback
  `./mlruns`).
- **Config is typed.** `load_config` returns a pydantic `Config`, so bad YAML
  fails before training starts. The data location is `data.uri` (local or remote).
- **Data arrives pre-split** into `train/`, `val/`, `test/` under `data.uri`.
  `preprocess()` converts formats only and never splits.
- **Task scope is image classification** for now (YOLO `-cls`, DINOv3 backbone
  + classifier head).

## Trainer methods

| Method | Meaning |
|---|---|
| `load()` | Build the model and load starting weights. Not data loading. |
| `preprocess()` | Convert the dataset to this model's format. |
| `train()` | Train. |
| `predict()` | Predict class labels. |

`evaluate()` is planned future work. Until then the `evaluate` task is generic
pipeline code: it calls `predict()` on the test split and computes accuracy / F1.

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
