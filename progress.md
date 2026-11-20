# Session Progress Log

## Current State

**Last Updated:** 2026-11-18
**Branch:** feature/load-config
**Active Feature:** none (feat-001 and feat-002 done; next is feat-003)

## Status

### What's Done

- [x] Train flow runs end to end for `logistic_regression` on `data/cifar10-mini`:
      load_config -> preprocess (80/10/10 manifest) -> train -> evaluate -> register
- [x] `register` logs flattened YAML params and accuracy/precision/recall/f1 to MLflow
- [x] feat-001: the train flow opens one MLflow run up front (inline
      `with mlflow.start_run(...)` in `pipeline/flow.py`, params logged before training);
      trainers can log metrics during `train()`; `register(metrics, trainer)` logs into
      the active run
- [x] Harness initialised: `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`

### What's In Progress

- [ ] Nothing in progress

### What's Next

1. feat-003: lazy-import trainers in `model_factory`
2. feat-004: tasks pass artifacts (largest change, last)

## Blockers / Risks

- [ ] `data/` is gitignored, so a clean checkout has no `data/cifar10-mini`; `init.sh`
      then skips the end-to-end step and only checks imports
- [ ] The `retrain` flow is still a stub (`NotImplementedError`); champion alias and
      promotion are not built
- [ ] `DinoTrainer` / `YoloTrainer` `log_model()` and `load()` are stubs

## Decisions Made

- **Data arrives unsplit**: `preprocess` splits into a manifest, then calls `trainer.preprocess()`
- **Hyperparameters are a free-form `params:` dict** in the YAML, validated by the trainer
- **ROC/AUC removed for now** to keep the code simple; `predict_proba()` went with them
- **`Trainer.load(model_uri=None)`** (agreed 2026-11-18): `None` = fresh/pretrained;
  otherwise an MLflow model URI. The caller resolves the champion URI, trainers never
  query the registry. `log_model()` takes no arguments and returns the model URI.
- **One MLflow run per flow** (2026-11-19): the train flow sets the tracking URI and
  experiment, then wraps every step after `load_config` in
  `with mlflow.start_run(run_name=cfg.model.name):` directly in the flow body (not in a
  Prefect task), so the run stays active in the flow's thread and is ended FINISHED or
  FAILED by the `with`. No hand-written try/except. `register(metrics, trainer)` no
  longer takes `cfg`. The first version's `tasks.start_run` helper was removed.
- **`get_trainer(config)`** (2026-11-19, user change): takes only the config and reads
  `config.model.name`.
- **Let the second model drive interface changes**: implement YOLO/HF and fix what it
  exposes, in the order of `feature_list.json`, instead of redesigning up front

## Evidence of Completion

- [x] `./init.sh` green on 2026-11-18: train flow logged params, metrics and a model;
      the model reloads via `mlflow.pyfunc.load_model` and via `trainer.load(model_uri)`
- [x] feat-001 `./init.sh` green on 2026-11-19: exactly 1 FINISHED run holds params,
      `train_accuracy` (logged inside `train()`), the final metrics and the model

## Notes for Next Session

- `init.sh` uses a temporary Prefect server and a throwaway MLflow database; it does not
  need `start.sh` and does not write to `mlflow.db`
- `init.sh` sometimes prints Prefect `EventsWorker` / "Cannot put items in a stopped
  service instance" errors at shutdown of the temporary server; they are noise and do
  not fail the script
- CLAUDE.md rule still applies: ask before fixing Task / `Trainer` method signatures
