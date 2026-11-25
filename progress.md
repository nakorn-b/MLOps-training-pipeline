# Session Progress Log

## Current State

**Last Updated:** 2026-11-25
**Branch:** see `git branch` (feature branches merged via PRs)
**Active Feature:** none (feat-006 done; next is feat-007, then feat-003)

## Status

### What's Done

- [x] Train flow runs end to end for `logistic_regression` on `data/cifar10-mini`:
      load_config -> preprocess (80/10/10 manifest) -> train -> evaluate -> register
- [x] `register` logs flattened YAML params and accuracy/precision/recall/f1 to MLflow
- [x] feat-001: the train flow opens one MLflow run up front (inline
      `with mlflow.start_run(...)` in `pipeline/flow.py`, params logged before training);
      trainers can log metrics during `train()`; `register(metrics, trainer)` logs into
      the active run
- [x] feat-005: `evaluate` also returns `test_roc_auc` (macro OvR) and logs one-vs-rest
      ROC curves as `roc_curve.png` to the active run; `init.sh` asserts both
- [x] Champion gating in `register`: every run registers a version under `model.registry_name`;
      the new version becomes `@champion` only if its `test_f1` beats the champion's
      (with no champion, all versions are compared and the best is crowned)
- [x] feat-006: `retrain` fine-tunes **any** MLflow model, not only the champion. The URI
      comes from `model.uri` in the YAML or `--model-uri` / the `model_uri` flow parameter
      (parameter wins); neither set -> error before any run opens.
      `LogisticRegressionTrainer.load(uri)` uses `warm_start=True` so it is a real fine-tune
- [x] Harness initialised: `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`

### What's In Progress

- [ ] Nothing in progress

### What's Next

1. feat-007: retrain gating on the new test split (score the champion on the same split)
2. feat-003: lazy-import trainers in `model_factory`
3. feat-004: tasks pass artifacts (largest change, last)

## Blockers / Risks

- [ ] `data/` is gitignored, so a clean checkout has no `data/cifar10-mini`; `init.sh`
      then skips the end-to-end step and only checks imports
- [ ] Retrain gating compares against the champion's *logged* `test_f1`, measured on the
      champion's own test split; after retraining on new data that comparison is invalid
      (feat-007)
- [ ] `DinoTrainer` / `YoloTrainer` `log_model()` and `load()` are stubs

## Decisions Made

- **Data arrives unsplit**: `preprocess` splits into a manifest, then calls `trainer.preprocess()`
- **Hyperparameters are a free-form `params:` dict** in the YAML, validated by the trainer
- **ROC/AUC is back** (2026-11-20): `Trainer` has `classes: Sequence` (column order of
  `predict_proba()`) and an abstract `predict_proba()` (no arguments, shape
  (n_test, n_classes)). `evaluate` returns `test_roc_auc` (macro one-vs-rest
  `roc_auc_score`) and logs one figure of per-class OvR ROC curves as `roc_curve.png`
  to the active MLflow run. `predict()` is unchanged.
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
- **Retrain fine-tunes any model by URI** (2026-11-25, user decision): `model.uri` in the
  YAML (`ModelConfig.uri`, optional) and `--model-uri` / `retrain(config_path,
  model_uri=None, dataset_uri=None)`; CLI/flow parameter overrides YAML; neither -> error.
  Overrides are applied to `cfg` before params are logged. `tasks.train(trainer,
  model_uri=None)` passes it to `Trainer.load()`.
- **Let the second model drive interface changes**: implement YOLO/HF and fix what it
  exposes, in the order of `feature_list.json`, instead of redesigning up front

## Evidence of Completion

- [x] `./init.sh` green on 2026-11-18: train flow logged params, metrics and a model;
      the model reloads via `mlflow.pyfunc.load_model` and via `trainer.load(model_uri)`
- [x] feat-001 `./init.sh` green on 2026-11-19: exactly 1 FINISHED run holds params,
      `train_accuracy` (logged inside `train()`), the final metrics and the model

- [x] feat-006 `./init.sh` green on 2026-11-25: retrain without a URI fails with no run
      opened; `--model-uri models:/<registry>/1` finishes, logs `params.model.uri`, registers v2

## Notes for Next Session

- `init.sh` uses a temporary Prefect server and a throwaway MLflow database; it does not
  need `start.sh` and does not write to `mlflow.db`
- `init.sh` sometimes prints Prefect `EventsWorker` / "Cannot put items in a stopped
  service instance" errors at shutdown of the temporary server; they are noise and do
  not fail the script
- CLAUDE.md rule still applies: ask before fixing Task / `Trainer` method signatures
