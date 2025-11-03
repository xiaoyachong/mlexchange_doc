# Complete Migration Guide: DVC to MLflow

## Table of Contents
1. [Overview](#overview)
2. [What's Changing](#whats-changing)
3. [Detailed Code Changes](#detailed-code-changes)
4. [HTML Report Generation](#html-report-generation)
5. [Dashboard Integration](#dashboard-integration)
6. [Testing the Migration](#testing-the-migration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide will help you migrate from DVClive to MLflow for experiment tracking while maintaining the same HTML report functionality for your Dash application.

### Current Architecture (with DVC)
```
train.py → DVClive → dvc.html → Dash Callback → Display
                  ↓
              MLflow (model only)
```

### New Architecture (MLflow only)
```
train.py → MLflow → report.html → Dash Callback → Display
              ↓
          (metrics + model + artifacts)
```

---

## What's Changing

### Files to Modify
1. **`src/train.py`** - Remove DVClive, add MLflow logger and HTML generation
2. **`src/tune.py`** - Same changes as train.py
3. **`requirements.txt`** or **`pyproject.toml`** - Remove dvclive dependency

### What Stays the Same
- ✅ **Callback code** (`src/callbacks/execute.py`) - NO CHANGES
- ✅ **File path** - Still `{WRITE_DIR}/{USER}/models/{job_id}/report.html`
- ✅ **Display method** - Still uses `html.Iframe(srcDoc=...)`
- ✅ **MLflow server setup** - Already configured

---

## Detailed Code Changes

### 1. Remove DVClive Dependencies

**Before (requirements.txt or pyproject.toml):**
```txt
dvclive
dvclive[lightning]
```

**After:**
```txt
# Remove dvclive entries
# Add these if not already present:
plotly>=5.0.0
kaleido>=0.2.1  # For static image export (optional)
```

---

### 2. Modify `src/train.py`

#### 2.1 Update Imports

**REMOVE these imports:**
```python
from dvclive import Live
from dvclive.lightning import DVCLiveLogger
```

**ADD these imports:**
```python
from pytorch_lightning.loggers import MLFlowLogger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

#### 2.2 Replace Training Section

**BEFORE (around line 65-95):**
```python
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
    model = Autoencoder(...)
    model.define_save_loss_dir(model_dir)

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training time: {time.time()-start}")

    # Log hyperparameters
    mlflow.log_params(train_parameters.dict())
    # Save model to MLflow
    mlflow.pytorch.log_model(
        model, "model", registered_model_name=io_parameters.uid_save
    )
```

**AFTER:**
```python
# Set up MLflow logger
mlflow_logger = MLFlowLogger(
    experiment_name=io_parameters.uid_save,
    tracking_uri=io_parameters.mlflow_uri,
    run_id=run_id  # Use the existing run ID from the context
)

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
    logger=mlflow_logger,  # Use MLflow logger instead of DVClive
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
training_time = time.time() - start
logger.info(f"Training time: {training_time:.2f} seconds")

# Log hyperparameters to MLflow
mlflow.log_params(train_parameters.dict())
mlflow.log_metric("training_time_seconds", training_time)

# Generate HTML report (replaces DVC report)
generate_training_report(
    model=model,
    model_dir=model_dir,
    train_parameters=train_parameters,
    io_parameters=io_parameters,
    run_id=run_id,
    training_time=training_time,
    input_channels=input_channels,
    width=width,
    height=height
)

# Save model to MLflow
mlflow.pytorch.log_model(
    model, "model", registered_model_name=io_parameters.uid_save
)
logger.info(
    f"Training complete. Model saved to MLflow with model name: {io_parameters.uid_save}"
)
```

---

## HTML Report Generation

### 3. Add Report Generation Function

Add this function to `src/train.py` (before the `if __name__ == "__main__":` block):

```python
def generate_training_report(
    model,
    model_dir,
    train_parameters,
    io_parameters,
    run_id,
    training_time,
    input_channels,
    width,
    height
):
    """
    Generate an interactive HTML report for training results.
    This replaces the DVClive HTML report functionality.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    logger.info("Generating training report...")
    
    epochs = list(range(len(model.train_loss_summary)))
    
    # Calculate learning rate schedule
    learning_rates = [
        train_parameters.learning_rate * 
        (train_parameters.gamma ** (e // train_parameters.step_size))
        for e in epochs
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training & Validation Loss',
            'Learning Rate Schedule',
            'Final Loss Comparison',
            'Loss Improvement Over Time'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Training and Validation Loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=model.train_loss_summary,
            name='Train Loss',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    if model.validation_loss_summary and len(model.validation_loss_summary) > 0:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=model.validation_loss_summary,
                name='Validation Loss',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # 2. Learning Rate Schedule
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=learning_rates,
            name='Learning Rate',
            mode='lines+markers',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    # 3. Final Loss Comparison (Bar Chart)
    final_train_loss = model.train_loss_summary[-1] if model.train_loss_summary else 0
    final_val_loss = model.validation_loss_summary[-1] if model.validation_loss_summary else 0
    
    fig.add_trace(
        go.Bar(
            x=['Train Loss', 'Validation Loss'],
            y=[final_train_loss, final_val_loss],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f'{final_train_loss:.6f}', f'{final_val_loss:.6f}'],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Loss Improvement (percentage reduction from epoch 0)
    if model.train_loss_summary:
        initial_train = model.train_loss_summary[0]
        train_improvement = [
            ((initial_train - loss) / initial_train) * 100 
            for loss in model.train_loss_summary
        ]
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_improvement,
                name='Train Loss Improvement (%)',
                mode='lines+markers',
                line=dict(color='#d62728', width=2),
                marker=dict(size=6),
                fill='tozeroy'
            ),
            row=2, col=2
        )
    
    if model.validation_loss_summary and len(model.validation_loss_summary) > 0:
        initial_val = model.validation_loss_summary[0]
        val_improvement = [
            ((initial_val - loss) / initial_val) * 100 
            for loss in model.validation_loss_summary
        ]
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_improvement,
                name='Val Loss Improvement (%)',
                mode='lines+markers',
                line=dict(color='#9467bd', width=2),
                marker=dict(size=6)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log")
    
    fig.update_xaxes(title_text="Metric", row=2, col=1)
    fig.update_yaxes(title_text="Loss Value", row=2, col=1)
    
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="Improvement (%)", row=2, col=2)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"Training Report: {io_parameters.uid_save}",
        title_font_size=20,
        hovermode='x unified'
    )
    
    # Create HTML content with metadata
    best_train_loss = min(model.train_loss_summary) if model.train_loss_summary else 0
    best_val_loss = min(model.validation_loss_summary) if model.validation_loss_summary else 0
    
    metadata_html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
        <h1 style="color: #333;">Training Report</h1>
        <h2 style="color: #666;">Experiment: {io_parameters.uid_save}</h2>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Training Time:</strong> {training_time:.2f} seconds</p>
        
        <h3 style="color: #333; margin-top: 30px;">Model Configuration</h3>
        <table style="border-collapse: collapse; width: 100%; background-color: white;">
            <tr style="background-color: #e0e0e0;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Parameter</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Value</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Latent Dimension</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.latent_dim}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Depth</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.depth}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Base Channel Size</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.base_channel_size}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Input Channels</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{input_channels}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Input Size</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{width} × {height}</td>
            </tr>
        </table>
        
        <h3 style="color: #333; margin-top: 30px;">Training Configuration</h3>
        <table style="border-collapse: collapse; width: 100%; background-color: white;">
            <tr style="background-color: #e0e0e0;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Parameter</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Value</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Learning Rate</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.learning_rate}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Batch Size</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.batch_size}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Number of Epochs</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.num_epochs}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Optimizer</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.optimizer.value}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Loss Function</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.criterion.value}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">LR Decay (Gamma)</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.gamma}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">LR Step Size</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{train_parameters.step_size}</td>
            </tr>
        </table>
        
        <h3 style="color: #333; margin-top: 30px;">Training Results</h3>
        <table style="border-collapse: collapse; width: 100%; background-color: white;">
            <tr style="background-color: #e0e0e0;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Metric</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Value</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Final Train Loss</td>
                <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{final_train_loss:.6f}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Final Validation Loss</td>
                <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{final_val_loss:.6f}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Best Train Loss</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #2ca02c; font-weight: bold;">{best_train_loss:.6f}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="border: 1px solid #ddd; padding: 8px;">Best Validation Loss</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #2ca02c; font-weight: bold;">{best_val_loss:.6f}</td>
            </tr>
        </table>
        
        <h3 style="color: #333; margin-top: 30px;">Training Curves</h3>
    </div>
    """
    
    # Get the Plotly HTML
    plot_html = fig.to_html(include_plotlyjs='cdn', div_id='training-plots')
    
    # Combine metadata and plot
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Training Report - {io_parameters.uid_save}</title>
        <style>
            body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            {metadata_html}
            {plot_html}
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = f"{model_dir}/report.html"
    with open(report_path, "w") as f:
        f.write(full_html)
    
    logger.info(f"HTML report saved to {report_path}")
    
    # Log to MLflow as artifact
    mlflow.log_artifact(report_path)
    
    # Also save plots as static images (optional, for quick preview)
    try:
        static_plot_path = f"{model_dir}/training_curves.png"
        fig.write_image(static_plot_path, width=1400, height=900)
        mlflow.log_artifact(static_plot_path)
        logger.info(f"Static plot saved to {static_plot_path}")
    except Exception as e:
        logger.warning(f"Could not save static image (kaleido may not be installed): {e}")
    
    # Save metrics as CSV for additional analysis
    try:
        import pandas as pd
        metrics_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': model.train_loss_summary,
            'validation_loss': model.validation_loss_summary if model.validation_loss_summary else [None] * len(epochs),
            'learning_rate': learning_rates
        })
        metrics_csv_path = f"{model_dir}/training_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(metrics_csv_path)
        logger.info(f"Training metrics CSV saved to {metrics_csv_path}")
    except Exception as e:
        logger.warning(f"Could not save metrics CSV: {e}")
```

---

## Dashboard Integration

### 4. Callback Code (NO CHANGES REQUIRED)

Your existing callback in `src/callbacks/execute.py` remains **exactly the same**:

```python
@callback(
    Output("stats-card-body", "children"),
    Output("show-plot", "is_open"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "show-training-stats",
            "aio_id": "data-clinic-jobs",
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "job-id",
            "aio_id": "data-clinic-jobs",
        },
        "children",
    ),
    prevent_initial_call=True,
)
def show_training_stats(show_stats_n_clicks, job_id):
    """Display training statistics from HTML report."""
    if not show_stats_n_clicks or not job_id:
        raise PreventUpdate
    
    try:
        # Get the training job ID
        children_job_ids = get_children_flow_run_ids(job_id)
        child_job_id = children_job_ids[0]  # training job
        
        # Load the HTML report (same path as before)
        expected_report_path = f"{WRITE_DIR}/{USER}/models/{child_job_id}/report.html"
        
        with open(expected_report_path, "r") as f:
            report_html = f.read()
        
        return (
            html.Iframe(
                srcDoc=report_html,
                style={"width": "100%", "height": "600px", "border": "none"}
            ),
            True,
        )
    
    except Exception as e:
        logger.error(f"Error loading training report: {e}")
        return (
            html.Div(
                f"Error loading training report: {str(e)}",
                style={"color": "red", "padding": "20px"}
            ),
            True,
        )
```

**Key Points:**
- ✅ No changes to the callback function
- ✅ Same file path: `{WRITE_DIR}/{USER}/models/{child_job_id}/report.html`
- ✅ Same display mechanism: `html.Iframe(srcDoc=...)`
- ✅ The callback doesn't know or care whether DVC or MLflow generated the HTML

---

### 5. Optional: Enhanced Callback with MLflow Fallback

If you want to add a fallback to download the report from MLflow when the local file is missing:

```python
import mlflow
from mlflow.tracking import MlflowClient

@callback(
    Output("stats-card-body", "children"),
    Output("show-plot", "is_open"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "show-training-stats",
            "aio_id": "data-clinic-jobs",
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "job-id",
            "aio_id": "data-clinic-jobs",
        },
        "children",
    ),
    prevent_initial_call=True,
)
def show_training_stats(show_stats_n_clicks, job_id):
    """Display training statistics from HTML report with MLflow fallback."""
    if not show_stats_n_clicks or not job_id:
        raise PreventUpdate
    
    try:
        # Get the training job ID
        children_job_ids = get_children_flow_run_ids(job_id)
        child_job_id = children_job_ids[0]  # training job
        
        # Try to load from local file system first
        expected_report_path = f"{WRITE_DIR}/{USER}/models/{child_job_id}/report.html"
        
        if os.path.exists(expected_report_path):
            logger.info(f"Loading report from local file: {expected_report_path}")
            with open(expected_report_path, "r") as f:
                report_html = f.read()
        else:
            # Fallback: Download from MLflow
            logger.info(f"Local report not found, downloading from MLflow for run: {child_job_id}")
            
            try:
                client = MlflowClient(tracking_uri=MLFLOW_URI)
                
                # Download the artifact
                artifact_path = client.download_artifacts(
                    run_id=child_job_id,
                    path="report.html"
                )
                
                with open(artifact_path, "r") as f:
                    report_html = f.read()
                
                logger.info(f"Successfully downloaded report from MLflow")
                
            except Exception as mlflow_error:
                logger.error(f"Error downloading from MLflow: {mlflow_error}")
                raise FileNotFoundError(
                    f"Report not found locally or in MLflow for job {child_job_id}"
                )
        
        return (
            html.Iframe(
                srcDoc=report_html,
                style={"width": "100%", "height": "600px", "border": "none"}
            ),
            True,
        )
    
    except Exception as e:
        logger.error(f"Error loading training report: {e}")
        return (
            html.Div([
                html.H4("Error Loading Training Report", style={"color": "red"}),
                html.P(str(e)),
                html.P(f"Job ID: {job_id}"),
                html.P(f"Expected path: {expected_report_path if 'expected_report_path' in locals() else 'N/A'}")
            ], style={"padding": "20px"}),
            True,
        )
```

---

## Testing the Migration

### Step 1: Install Dependencies

```bash
# Remove DVClive
pip uninstall dvclive -y

# Install Plotly (if not already installed)
pip install plotly>=5.0.0

# Optional: Install kaleido for static image export
pip install kaleido>=0.2.1
```

### Step 2: Test Training Script

```bash
# Run a test training job
python src/train.py path/to/test_config.yaml
```

**What to verify:**
1. ✅ Training completes without DVClive errors
2. ✅ HTML report is generated at `models/{job_id}/report.html`
3. ✅ Report is logged to MLflow artifacts
4. ✅ Metrics appear in MLflow UI
5. ✅ Model is registered in MLflow

### Step 3: Check MLflow UI

```bash
# Access MLflow UI (if running locally)
mlflow ui --host 0.0.0.0 --port 5000
```

Navigate to:
1. **Experiments** → Find your experiment
2. **Runs** → Click on the latest run
3. **Artifacts** → Verify `report.html` is present
4. **Metrics** → Verify `train_loss` and `val_loss` are logged
5. **Parameters** → Verify all hyperparameters are logged

### Step 4: Test Dashboard Callback

1. Navigate to your Dash application
2. Run a training job
3. Click "Show Training Stats"
4. Verify the HTML report displays correctly

**Expected behavior:**
- Interactive Plotly charts should be visible
- Tables with parameters and metrics should be formatted correctly
- Zoom, pan, and hover tooltips should work

---

## Modify `src/tune.py` (Similar Changes)

Apply the same pattern to `tune.py`:

**REMOVE:**
```python
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

with Live(model_dir, report="html") as live:
    trainer = pl.Trainer(..., logger=DVCLiveLogger(experiment=live))
```

**REPLACE WITH:**
```python
from pytorch_lightning.loggers import MLFlowLogger

# Set up MLflow for tuning
mlflow.set_tracking_uri(io_parameters.mlflow_uri)
mlflow.set_experiment(f"{io_parameters.uid_save}_tuning")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    
    mlflow_logger = MLFlowLogger(
        experiment_name=f"{io_parameters.uid_save}_tuning",
        tracking_uri=io_parameters.mlflow_uri,
        run_id=run_id
    )
    
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=tune_parameters.num_epochs,
        enable_progress_bar=False,
        profiler=tune_parameters.profiler.value if tune_parameters.profiler else None,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                save_last=True,
                filename="checkpoint_file",
                save_weights_only=True,
            )
        ],
        logger=mlflow_logger,
    )
    
    # Load and tune model
    model = Autoencoder.load_from_checkpoint(model_dir + "/last.ckpt")
    model.define_save_loss_dir(output_dir)
    model.optimizer = getattr(optim, tune_parameters.optimizer.value)
    criterion = getattr(nn, tune_parameters.criterion.value)
    model.criterion = criterion()
    model.learning_rate = tune_parameters.learning_rate
    model.gamma = tune_parameters.gamma
    model.step_size = tune_parameters.step_size
    
    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    tuning_time = time.time() - start
    
    logger.info(f"Tuning time: {tuning_time:.2f} seconds")
    
    # Log tuning parameters
    mlflow.log_params(tune_parameters.dict())
    mlflow.log_metric("tuning_time_seconds", tuning_time)
    
    # Generate report for tuning results
    generate_training_report(
        model=model,
        model_dir=output_dir,
        train_parameters=tune_parameters,  # Use tune_parameters
        io_parameters=io_parameters,
        run_id=run_id,
        training_time=tuning_time,
        input_channels=input_channels,
        width=width,
        height=height
    )
```

---

## Troubleshooting

### Issue 1: "No module named 'dvclive'"

**Cause:** DVClive is still being imported somewhere

**Solution:**
```bash
# Search for remaining imports
grep -r "from dvclive" src/
grep -r "import dvclive" src/

# Remove all DVClive imports
```

### Issue 2: HTML Report Not Found in Dashboard

**Cause:** Report path mismatch or report not generated

**Solution:**
1. Check if report was created:
   ```bash
   ls -la {WRITE_DIR}/{USER}/models/{job_id}/report.html
   ```

2. Verify the `model_dir` variable in `train.py` matches the expected path

3. Check MLflow artifacts as fallback:
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient()
   artifacts = client.list_artifacts(run_id=job_id)
   print([a.path for a in artifacts])
   ```

### Issue 3: Plots Not Interactive in Dashboard

**Cause:** Plotly JavaScript not loading

**Solution:**
Ensure the HTML includes Plotly CDN:
```python
plot_html = fig.to_html(include_plotlyjs='cdn')  # NOT 'directory'
```

### Issue 4: "Failed to write image, kaleido not installed"

**Cause:** Static image export failed (optional feature)

**Solution:**
```bash
# Install kaleido for static exports
pip install kaleido

# Or ignore this warning - static images are optional
```

### Issue 5: MLflow Metrics Not Appearing

**Cause:** PyTorch Lightning logger not configured correctly

**Solution:**
Ensure you're using `self.log()` in your model:
```python
# In model.py
def training_step(self, batch, batch_idx):
    loss = self._get_reconstruction_loss(batch)
    self.log("train_loss", loss, on_epoch=True)  # This logs to MLflow
    return loss
```

### Issue 6: Permission Errors When Writing Report

**Cause:** Directory doesn't exist or insufficient permissions

**Solution:**
```python
# Ensure directory exists
from pathlib import Path
Path(model_dir).mkdir(parents=True, exist_ok=True)
```

---

## Summary Checklist

### Code Changes
- [ ] Remove `dvclive` from `requirements.txt` or `pyproject.toml`
- [ ] Add `plotly>=5.0.0` to dependencies
- [ ] Update imports in `train.py` (remove DVClive, add MLFlowLogger)
- [ ] Replace DVClive logger with MLFlowLogger in `train.py`
- [ ] Add `generate_training_report()` function to `train.py`
- [ ] Call report generation after training
- [ ] Update `tune.py` with same changes

### Dashboard
- [ ] No changes needed to callback code
- [ ] (Optional) Add MLflow fallback to callback

### Testing
- [ ] Uninstall dvclive
- [ ] Install plotly
- [ ] Run test training job
- [ ] Verify HTML report is created
- [ ] Check MLflow UI for artifacts
- [ ] Test dashboard report display
- [ ] Verify interactive plots work

### Benefits of Migration
✅ **Single tool:** MLflow handles tracking, logging, and model registry  
✅ **Better UI:** MLflow provides powerful experiment comparison  
✅ **Interactive plots:** Plotly charts are more feature-rich than static DVC plots  
✅ **API access:** Query metrics and artifacts programmatically  
✅ **Scalability:** MLflow is designed for production ML workflows  
✅ **No breaking changes:** Dashboard callbacks remain unchanged  

---

## Additional Resources

- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **Plotly Python:** https://plotly.com/python/
- **PyTorch Lightning MLflow Integration:** https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.MLFlowLogger.html

---

*Last Updated: 2025-11-03*
