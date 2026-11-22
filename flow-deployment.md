## Work pool

Create a process work pool
```bash
prefect work-pool create --type process mlops-pipeline
```

Start workpool
```bash
prefect worker start --pool mlops-pipeline
```

Run deployment
```bash
prefect deployment run 'train/train_pipeline' --param config_path="configs/example.yaml"  
```