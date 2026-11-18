# Session Progress Log

## Current State

**Last Updated:** 2026-11-18
**Branch:** feature/load-config
**Active Feature:** none (feat-002 done; next is feat-001)

## Status

### What's Done

- [x] Train flow runs end to end for `logistic_regression` on `data/cifar10-mini`:
      load_config -> preprocess (80/10/10 manifest) -> train -> evaluate -> register
- [x] `register` logs flattened YAML params and accuracy/precision/recall/f1 to MLflow
- [x] Harness initialised: `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`

### What's In Progress

- [ ] Nothing in progress

### What's Next

1. feat-001: open the MLflow run at the start of the flow (then `register` stops calling `mlflow.start_run`)
2. feat-003: lazy-import trainers in `model_factory`
3. feat-004: tasks pass artifacts (largest change, last)

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
- **Let the second model drive interface changes**: implement YOLO/HF and fix what it
  exposes, in the order of `feature_list.json`, instead of redesigning up front

## Evidence of Completion

- [x] `./init.sh` green on 2026-11-18: train flow logged params, metrics and a model;
      the model reloads via `mlflow.pyfunc.load_model` and via `trainer.load(model_uri)`

## Notes for Next Session

- `init.sh` uses a temporary Prefect server and a throwaway MLflow database; it does not
  need `start.sh` and does not write to `mlflow.db`
- CLAUDE.md rule still applies: ask before fixing Task / `Trainer` method signatures
