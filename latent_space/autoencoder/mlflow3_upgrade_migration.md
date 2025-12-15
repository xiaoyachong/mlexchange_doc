# MLflow 3 Migration Guide for Your Project

Based on the official MLflow 3 documentation, here are the **critical changes** you need to make to your codebase.

## 🔴 **BREAKING CHANGE: Model Logging API**

The most significant change is in how you log models. The `artifact_path` parameter is deprecated in favor of `name`.

### **train.py - REQUIRED CHANGES:**

**OLD (MLflow 2.x):**
```python
with mlflow.start_run() as run:
    # ... training code ...
    
    mlflow.log_params(train_parameters.dict())
    mlflow.pytorch.log_model(
        model, 
        "model",  # This was artifact_path
        registered_model_name=io_parameters.uid_save
    )
```

**NEW (MLflow 3.x):**
```python
with mlflow.start_run() as run:
    # ... training code ...
    
    mlflow.log_params(train_parameters.dict())
    
    # Option 1: Log model within a run (recommended for your workflow)
    mlflow.pytorch.log_model(
        model,
        name=io_parameters.uid_save,  # Changed from artifact_path to name
        registered_model_name=io_parameters.uid_save  # Still works for registry
    )
    
    # Option 2: Log model without run (new in MLflow 3)
    # mlflow.pytorch.log_model(
    #     model,
    #     name=io_parameters.uid_save
    # )
```

## 🔴 **BREAKING CHANGE: Model Artifacts Storage**

Models are now stored in a separate models artifacts location instead of run artifacts. This affects how you retrieve models.

### **inference.py - REQUIRED CHANGES:**

**OLD (MLflow 2.x):**
```python
model = mlflow.pytorch.load_model(f"models:/{model_name}/latest")
```

**NEW (MLflow 3.x):**
```python
# The URI format remains the same, but the underlying storage changed
# You should explicitly handle the model loading

if hasattr(io_parameters, "mlflow_model") and io_parameters.mlflow_model:
    model_name = io_parameters.mlflow_model
else:
    model_name = io_parameters.uid_retrieve

logger.info(f"Loading latest model from MLflow registry: {model_name}")

# Load from registry (preferred method)
model = mlflow.pytorch.load_model(f"models:/{model_name}/latest")

# Alternative: Load by run_id if you track it
# model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
```

## 📋 **Complete Updated Code**

### **train.py (lines 92-142):**

```python
# Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Set seed
    if train_parameters.seed:
        seed = train_parameters.seed
    else:
        seed = SEED
    pl.seed_everything(seed)
    logger.info("Seed: " + str(seed))

    # Set device
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device:" + str(device))

    # Set target size
    if train_parameters.target_width * train_parameters.target_height > 0:
        target_size = (
            train_parameters.target_width,
            train_parameters.target_height,
        )
    else:
        target_size = None

    # Get dataloaders
    logger.info(f"Number of workers: {train_parameters.num_workers}")
    [train_loader, val_loader], (input_channels, width, height) = (
        get_train_dataloaders(
            io_parameters.data_uris,
            io_parameters.root_uri,
            io_parameters.data_type,
            train_parameters.batch_size,
            train_parameters.num_workers,
            train_parameters.shuffle,
            target_size,
            train_parameters.horz_flip_prob,
            train_parameters.vert_flip_prob,
            train_parameters.brightness,
            train_parameters.contrast,
            train_parameters.saturation,
            train_parameters.hue,
            train_parameters.val_pct,
            train_parameters.augm_invariant,
            train_parameters.log,
            data_tiled_api_key=io_parameters.data_tiled_api_key,
            detector_uri=io_parameters.detector_uri,
            detector_source=io_parameters.detector_source,
            detector_tiled_api_key=io_parameters.detector_tiled_api_key,
        )
    )

    # Set up model directory
    model_dir = Path(f"{io_parameters.models_dir}/{io_parameters.uid_save}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set up dvclive
    with Live(model_dir, report="html") as live:
        trainer = pl.Trainer(
            default_root_dir=model_dir,
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=train_parameters.num_epochs,
            enable_progress_bar=False,
            profiler=train_parameters.profiler,
            callbacks=[
                ModelCheckpoint(
                    dirpath=model_dir,
                    save_last=True,
                    filename="checkpoint_file",
                    save_weights_only=True,
                )
            ],
            logger=DVCLiveLogger(experiment=live),
        )

        # Set up model
        model = Autoencoder(
            base_channel_size=train_parameters.base_channel_size,
            depth=train_parameters.depth,
            latent_dim=train_parameters.latent_dim,
            num_input_channels=input_channels,
            optimizer=train_parameters.optimizer,
            criterion=train_parameters.criterion,
            learning_rate=train_parameters.learning_rate,
            step_size=train_parameters.step_size,
            gamma=train_parameters.gamma,
            width=width,
            height=height,
        )
        model.define_save_loss_dir(model_dir)

        start = time.time()
        trainer.fit(model, train_loader, val_loader)
        logger.info(f"Training time: {time.time()-start}")

        # Log hyperparameters
        mlflow.log_params(train_parameters.dict())
        
        # CHANGED: Use 'name' instead of artifact_path
        mlflow.pytorch.log_model(
            model,
            name=io_parameters.uid_save,  # This is the key change
            registered_model_name=io_parameters.uid_save
        )
        logger.info(
            f"Training complete. Model saved to MLflow with model name: {io_parameters.uid_save}"
        )
```

## 🔍 **What Changed:**

1. **`artifact_path` → `name`**: The first positional argument in `log_model()` is now `name` instead of `artifact_path`
2. **Model storage location**: Models are stored separately from run artifacts (internal change, no code impact)
3. **`mlflow.start_run()` is optional**: You can now log models without starting a run, but keeping it is fine for your workflow

## ✅ **What Stays the Same:**

1. `mlflow.set_experiment()` - still works
2. `mlflow.set_tracking_uri()` - still works
3. `mlflow.log_params()` - still works
4. Model loading URIs - still works (`models:/`, `runs:/`)
5. Authentication via environment variables - still works

## 🎯 **Minimal Required Changes:**

### **File: src/train.py (around line 141)**

**BEFORE:**
```python
mlflow.pytorch.log_model(
    model, 
    "model", 
    registered_model_name=io_parameters.uid_save
)
```

**AFTER:**
```python
mlflow.pytorch.log_model(
    model, 
    name=io_parameters.uid_save, 
    registered_model_name=io_parameters.uid_save
)
```

That's the **only critical change** you must make! The rest of your code is compatible with MLflow 3.

## 📚 **Optional Enhancements (Recommended)**

### **Add Model Signature (train.py)**

Adding signatures helps with model validation and documentation:

```python
# After training, before logging the model
sample_input = next(iter(train_loader))[0][:1]  # Get one sample
sample_output = model(sample_input).detach()

mlflow.pytorch.log_model(
    model,
    name=io_parameters.uid_save,
    registered_model_name=io_parameters.uid_save,
    signature=mlflow.models.infer_signature(
        sample_input.cpu().numpy(),
        sample_output.cpu().numpy()
    ),
    input_example=sample_input.cpu().numpy()
)
```

### **Use Model Aliases (New in MLflow 3)**

Model aliases replace the old staging system:

```python
from mlflow import MlflowClient

# After logging the model
client = MlflowClient()
latest_version = client.get_latest_versions(io_parameters.uid_save)[0].version
client.set_registered_model_alias(
    name=io_parameters.uid_save,
    alias="champion",  # or "production", "staging", etc.
    version=latest_version
)
```

Then in inference.py, you can load by alias:
```python
model = mlflow.pytorch.load_model(f"models:/{model_name}@champion")
```

### **Enable Autologging (Optional)**

MLflow 3 has improved autologging for PyTorch Lightning:

```python
# Add to train.py before creating the trainer
mlflow.pytorch.autolog(
    log_every_n_epoch=1,
    log_models=False,  # We'll log manually with signature
    disable=False
)
```

## 🚨 **Removed Features in MLflow 3**

These features are no longer supported:
- MLflow Recipes
- Flavors: `fastai`, `mleap`
- AI gateway client APIs (use deployments APIs instead)

Your code doesn't use any of these, so you're good!

## 📌 **Migration Checklist**

- [ ] Update `train.py` line ~141: Change `artifact_path` to `name`
- [ ] Test training with MLflow 3
- [ ] Test inference with MLflow 3
- [ ] (Optional) Add model signatures
- [ ] (Optional) Implement model aliases
- [ ] Update MLflow server to version 3.x
- [ ] Update MLflow client library: `pip install mlflow>=3.0.0`

## 🔗 **Additional Resources**

- [MLflow 3 Migration Guide](https://mlflow.org/docs/latest/migration-guide.html)
- [MLflow 3 Breaking Changes](https://mlflow.org/docs/latest/breaking-changes.html)
- [MLflow PyTorch Documentation](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)

---

**Summary**: The main change is replacing `artifact_path` with `name` in your `log_model()` call. Everything else in your code is already compatible with MLflow 3!
