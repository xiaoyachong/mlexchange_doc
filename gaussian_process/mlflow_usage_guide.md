# MLflow Usage Guide for AARDVARK

## Overview
This guide explains how to use MLflow to track, analyze, and compare experiments in the AARDVARK system.

## Accessing MLflow UI

Open your browser and navigate to:
```
http://localhost:5000
```

## Understanding the Interface

### 1. Experiments Page

When you first open MLflow, you'll see a list of experiments:
- Each AARDVARK experiment has a unique name: `AARDVARK_Experiment_1`, `AARDVARK_Experiment_2`, etc.
- Click on an experiment name to see all training iterations (runs)

### 2. Runs Table

Inside an experiment, you'll see a table of runs (training iterations):

| Column | Description |
|--------|-------------|
| **Start Time** | When this training iteration started |
| **Run Name** | Format: `iteration_0`, `iteration_1`, etc. |
| **Duration** | How long the training took |
| **Parameters** | Click to see all parameters |
| **Metrics** | Click to see all logged metrics |

### 3. Run Details Page

Click on any run to see detailed information:

#### Parameters Tab
Shows configuration used for this iteration:
```
n_measurements: 25
n_features: 65536
input_bounds_x: [0.0, 10.0]
input_bounds_y: [-5.0, 5.0]
device: cuda:0
iteration: 5
intensity_factor: 1
umap_factor: 1
gp_learning_rate: 0.1
gp_kernel_type: RBFKernel
umap_n_components: 3
```

#### Metrics Tab
Shows performance metrics:
```
training_time_ms: 2543.21
umap_fit_time_sec: 1.234
gp_fit_time_sec: 0.987
gp_final_loss: -125.43
total_data_points: 25
prediction_time_sec: 0.543
mean_uncertainty: 0.234
max_uncertainty: 0.876
mean_predicted_intensity: 0.567
```

#### Artifacts Tab
Contains saved files:
- `gp_model/` - PyTorch model checkpoint
- `model/likelihood_state_1_5.pth` - Likelihood parameters
- `suggestions/suggested_positions_1_5.json` - Next measurement locations

## Comparing Experiments

### Compare Across Iterations

1. Select multiple runs (use checkboxes)
2. Click "Compare" button
3. View side-by-side comparison of:
   - Parameters
   - Metrics
   - Charts

### Visualizing Metrics

MLflow automatically creates charts for metrics:

**Training Time Trends**
- X-axis: Iteration number
- Y-axis: Training time (ms)
- Shows how training time changes as data accumulates

**Uncertainty Evolution**
- Track `mean_uncertainty` over iterations
- See if model becomes more confident

**Loss Convergence**
- View `gp_loss` during training (step-wise)
- Check if GP is converging properly

### Creating Custom Charts

1. Click "Chart" tab in experiment view
2. Select metrics to plot (e.g., `mean_uncertainty` vs `iteration`)
3. Choose chart type (line, scatter, bar)
4. Save chart for later reference

## Using Logged Data

### Download Model Checkpoints

1. Navigate to a run's Artifacts tab
2. Click on `gp_model/` folder
3. Download `model.pth` and other files
4. Load in Python:

```python
import mlflow
import torch

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load model
run_id = "your-run-id-here"
model = mlflow.pytorch.load_model(f"runs:/{run_id}/gp_model")

# Use model for predictions
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

### Access Suggested Positions

Download `suggested_positions_X_Y.json`:

```json
{
  "positions": [
    [1.5, 2.3],
    [0.8, 1.9],
    [3.2, 0.7]
  ],
  "acquisition_values": [0.876, 0.654, 0.543],
  "predicted_intensities": [0.789, 0.456, 0.234]
}
```

Use in analysis:
```python
import json

with open('suggested_positions_1_5.json', 'r') as f:
    data = json.load(f)

positions = data['positions']
acquisition = data['acquisition_values']

# Analyze which positions were most valuable
import matplotlib.pyplot as plt
plt.scatter(*zip(*positions), c=acquisition, cmap='viridis')
plt.colorbar(label='Acquisition Value')
plt.show()
```

## Analyzing Experiment Performance

### Key Questions to Ask

**1. Is the model training efficiently?**
- Check `training_time_ms` trend
- If increasing linearly, normal
- If increasing exponentially, may need optimization

**2. Is uncertainty decreasing?**
- Plot `mean_uncertainty` over iterations
- Should generally decrease as more data is collected
- Plateaus indicate well-explored regions

**3. Is the GP converging?**
- Check `gp_final_loss` across iterations
- Should stabilize after initial iterations
- Large fluctuations may indicate instability

**4. Are predictions getting better?**
- Compare `max_uncertainty` over time
- High values indicate unexplored regions
- Use to guide experiment continuation

### SQL Queries

MLflow stores data in SQLite. Query directly:

```python
import sqlite3
import pandas as pd

# Connect to MLflow database
conn = sqlite3.connect('/path/to/mlflow/mlflow.db')

# Query all metrics for an experiment
query = """
SELECT r.run_id, r.start_time, m.key, m.value
FROM runs r
JOIN metrics m ON r.run_id = m.run_id
WHERE r.experiment_id = '1'
"""

df = pd.read_sql(query, conn)
print(df)
```

## Model Registry

### Registering Best Models

1. Navigate to a run with good performance
2. Click on `gp_model` artifact
3. Click "Register Model" button
4. Choose model name: `AARDVARK_GP_Experiment_1`
5. Add version description

### Managing Model Versions

- **Stage**: Development → Staging → Production
- **Tags**: Add metadata (e.g., `best_uncertainty: 0.123`)
- **Description**: Document what makes this model special

### Using Registered Models

```python
import mlflow.pytorch

# Load production model
model = mlflow.pytorch.load_model(
    "models:/AARDVARK_GP_Experiment_1/Production"
)

# Or load specific version
model = mlflow.pytorch.load_model(
    "models:/AARDVARK_GP_Experiment_1/3"
)
```

## Best Practices

### 1. Use Descriptive Run Names
Edit run name after training:
```python
with mlflow.start_run(run_name=f"iteration_{i}_high_intensity"):
    # training code
```

### 2. Add Tags
Tag runs with important information:
```python
mlflow.set_tags({
    "sample_type": "graphene",
    "temperature": "300K",
    "photon_energy": "100eV"
})
```

### 3. Log Important Events
Add notes to runs:
```python
mlflow.log_param("notes", "Changed acquisition function")
```

### 4. Archive Old Experiments
Delete or archive completed experiments:
```bash
# Delete experiment
mlflow experiments delete --experiment-id 1

# Restore if needed
mlflow experiments restore --experiment-id 1
```

## Programmatic Access

### Python API

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("AARDVARK_Experiment_1")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.mean_uncertainty < 0.5",
    order_by=["metrics.mean_uncertainty ASC"]
)

# Get best run
best_run = runs.iloc[0]
print(f"Best run ID: {best_run.run_id}")
print(f"Uncertainty: {best_run['metrics.mean_uncertainty']}")
```

### REST API

```bash
# Get experiment info
curl http://localhost:5000/api/2.0/mlflow/experiments/get?experiment_id=1

# Search runs
curl -X POST http://localhost:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"]}'
```

## Exporting Data

### Export to CSV

```python
import mlflow
import pandas as pd

# Get all runs
client = mlflow.tracking.MlflowClient()
experiment_id = "1"
runs = client.search_runs(experiment_id)

# Convert to DataFrame
data = []
for run in runs:
    row = {
        'run_id': run.info.run_id,
        'start_time': run.info.start_time,
        **run.data.params,
        **run.data.metrics
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('experiment_results.csv', index=False)
```

### Export Models

```bash
# Export model as Python function
mlflow models build-docker \
  -m runs:/YOUR_RUN_ID/gp_model \
  -n my-model-image

# Serve model via REST API
mlflow models serve \
  -m runs:/YOUR_RUN_ID/gp_model \
  -p 5001
```

## Troubleshooting

### Can't Access MLflow UI

```bash
# Check if MLflow is running
docker compose ps mlflow

# Check logs
docker compose logs mlflow

# Restart MLflow
docker compose restart mlflow
```

### Missing Metrics

Check code for proper logging:
```python
# Ensure you're in an active run
with mlflow.start_run():
    mlflow.log_metric("my_metric", value)
```

### Large Artifact Storage

MLflow stores artifacts in `/mlflow/artifacts`. Monitor size:
```bash
# Check artifact storage size
docker compose exec mlflow du -sh /mlflow/artifacts
```

Clean up old artifacts manually if needed.

## Advanced Features

### Auto-logging

Enable automatic PyTorch logging:
```python
import mlflow.pytorch

mlflow.pytorch.autolog()

# Training happens automatically logged
model.fit(X, y)
```

### Parallel Runs

Track multiple experiments simultaneously:
```python
# Different threads/processes
with mlflow.start_run(run_name="config_A"):
    train_model_A()

with mlflow.start_run(run_name="config_B"):
    train_model_B()
```

### Custom Metrics

Log custom computed metrics:
```python
# Calculate information gain
info_gain = calculate_info_gain(predictions, ground_truth)
mlflow.log_metric("information_gain", info_gain)

# Log metric at specific step
for epoch in range(100):
    loss = train_step()
    mlflow.log_metric("loss", loss, step=epoch)
```

## Integration with Analysis Tools

### Jupyter Notebooks

```python
# In Jupyter
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# Interactive exploration
%load_ext mlflow
%mlflow experiment AARDVARK_Experiment_1
```

### TensorBoard

Export MLflow data to TensorBoard:
```bash
mlflow export \
  --experiment-id 1 \
  --output-dir tensorboard_logs \
  --format tensorboard
```

## Summary

MLflow provides comprehensive experiment tracking for AARDVARK:
- **Automatic logging** of all training parameters and metrics
- **Model versioning** for reproducibility
- **Artifact storage** for models and data files
- **Comparison tools** for optimization
- **Programmatic access** for custom analysis

Use MLflow to:
1. Monitor experiment progress in real-time
2. Compare different experimental conditions
3. Reproduce successful experiments
4. Share results with collaborators
5. Deploy best models for production use
