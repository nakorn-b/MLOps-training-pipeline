## Pipeline Flow

```mermaid
flowchart LR
    preprocess["<b>preprocess</b><br/>split 80/10/10<br/>+ model-specific prep"]
    train["<b>train</b><br/>load model<br/>+ fit on train split"]
    evaluate["<b>evaluate</b><br/>accuracy, precision,<br/>recall, f1 on test split"]
    register["<b>register</b><br/>log params + metrics<br/>to MLflow"]

    preprocess --> train --> evaluate --> register
```

The `train` flow in `pipeline/flow.py` runs these four tasks in order, after
reading the experiment YAML and building the trainer named in `model.name`.

| Step | Input | Output |
|---|---|---|
| `preprocess` | `data.uri`: a folder with one subfolder per class | a manifest (`path`, `label`, `split`); the trainer holds its prepared data |
| `train` | the trainer | a fitted model inside the trainer |
| `evaluate` | the trainer | a metrics dict |
| `register` | the config and the metrics | one MLflow run |
