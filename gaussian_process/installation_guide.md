# AARDVARK with MLflow Integration - Installation Guide

## Overview
This guide walks you through installing and running the AARDVARK autonomous experimental design system with MLflow experiment tracking.

## Prerequisites

### Required Software
- Docker (version 20.10+)
- Docker Compose (version 2.0+)
- NVIDIA GPU with CUDA support (optional but recommended)
- NVIDIA Container Toolkit (for GPU support)

### System Requirements
- At least 16GB RAM
- 50GB free disk space
- Linux or macOS (tested on Ubuntu 20.04+)

## Installation Steps

### 1. Download and Extract Files

Download all the modified files and place them in your project directory:

```
aardvark/
├── app/
│   ├── celery_workers/
│   │   ├── agents.py          # Modified with MLflow
│   │   ├── celery_app.py
│   │   └── tasks.py
│   ├── db/
│   │   ├── config.py          # Modified with MLflow URI
│   │   ├── database.py
│   │   ├── models.py
│   │   └── base.py
│   ├── maestro_api/
│   │   ├── maestro_app.py
│   │   └── maestro_messages.py
│   ├── dash_app.py
│   ├── main.py
│   └── server.py
├── compose.yaml               # Modified with MLflow service
├── celery_requirements.txt    # Modified with mlflow
├── server_requirements.txt
├── worker.Dockerfile
├── server.Dockerfile
├── .env                       # Create this file
└── README.md
```

### 2. Create Environment File

Create a `.env` file in the root directory:

```bash
# .env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=arpes_aardvark
POSTGRES_DB=aardvark_db
```

### 3. Build Docker Images

```bash
# Build all services
docker compose build

# This will take 10-20 minutes on first build
```

### 4. Start Services

```bash
# Start all services (database, Redis, MLflow, workers, etc.)
docker compose up -d

# Check that all services are running
docker compose ps
```

You should see these services running:
- `db` (PostgreSQL)
- `redis`
- `mlflow`
- `dash_app`
- `server`
- `celery_worker`

### 5. Verify Installation

#### Check MLflow
Open your browser and navigate to:
```
http://localhost:5000
```

You should see the MLflow UI with no experiments yet.

#### Check Dashboard
```
http://localhost:80/dashboard
```

You should see the Dash application interface.

#### Check Logs
```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f celery_worker
docker compose logs -f mlflow
```

## Running an Experiment

### Option 1: Using Fake LabVIEW Client

1. Install Python dependencies on your host machine:
```bash
pip install numpy zmq pydantic scipy matplotlib
```

2. Run the fake client:
```bash
python fake_labview.py
```

This will:
- Initialize an experiment
- Send simulated measurements
- Trigger the autonomous learning loop
- Show progress in the terminal

### Option 2: Real LabVIEW Integration

Configure your LabVIEW system to connect to:
```
Server: tcp://your-server-ip:5550
```

Follow the LabVIEW integration documentation for your specific setup.

## Monitoring Experiments

### MLflow UI

Navigate to `http://localhost:5000` to see:

1. **Experiments**: Listed by experiment ID
   - Click on an experiment to see all training iterations

2. **Runs**: Each iteration of the learning loop
   - View parameters (batch size, bounds, device)
   - View metrics (training time, losses, uncertainties)
   - Download artifacts (models, suggested positions)

3. **Models**: Versioned GP models
   - Download model checkpoints
   - View model metadata
   - Compare model performance

### Dashboard

Navigate to `http://localhost:80/dashboard` to see:
- Real-time measurement positions
- Uncertainty maps
- GP predictions
- UMAP visualizations
- Acquisition function heatmaps

## Key Metrics Logged to MLflow

### Training Phase (`tell()`)
- `training_time_ms`: Total time to train models
- `umap_fit_time_sec`: UMAP dimensionality reduction time
- `gp_fit_time_sec`: Gaussian Process training time
- `gp_loss`: GP loss during training (every 10 iterations)
- `gp_final_loss`: Final GP loss value
- `total_data_points`: Number of measurements used for training

### Prediction Phase (`ask()`)
- `prediction_time_sec`: Time to generate predictions
- `mean_uncertainty`: Average uncertainty across candidate points
- `max_uncertainty`: Maximum uncertainty value
- `mean_predicted_intensity`: Average predicted signal intensity
- `n_positions_suggested`: Number of next positions suggested
- `top_acquisition_value`: Highest acquisition function value

### Parameters Logged
- `n_measurements`: Number of measurements in current iteration
- `n_features`: Dimensionality of input spectra
- `input_bounds_x/y`: Spatial boundaries
- `device`: CPU or CUDA device
- `iteration`: Training iteration number
- `umap_n_components`: UMAP output dimensions (3)
- `gp_learning_rate`: GP optimizer learning rate
- `gp_kernel_type`: Kernel function (RBFKernel)

### Artifacts Saved
- **GP Model**: Full PyTorch model checkpoint
- **Likelihood State**: GP likelihood parameters
- **Suggested Positions**: JSON file with next measurement locations
- **Acquisition Values**: Values for each suggested position
- **Code**: Source code used for training

## Troubleshooting

### MLflow Not Starting
```bash
# Check MLflow logs
docker compose logs mlflow

# Restart MLflow service
docker compose restart mlflow
```

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check worker logs
docker compose logs celery_worker | grep -i cuda
```

### Database Connection Issues
```bash
# Reset database
docker compose down -v
docker compose up -d db
# Wait for database to initialize
docker compose up -d
```

### Port Conflicts
If ports 80 or 5000 are already in use:

Edit `compose.yaml`:
```yaml
# Change MLflow port
mlflow:
  ports:
    - "5001:5000"  # Use 5001 instead of 5000

# Change dashboard port
dash_app:
  ports:
    - "8080:80"    # Use 8080 instead of 80
```

## Data Persistence

### MLflow Data
MLflow data is stored in Docker volume `mlflow-data`:
```bash
# Backup MLflow data
docker run --rm -v aardvark_mlflow-data:/data -v $(pwd):/backup ubuntu tar czf /backup/mlflow-backup.tar.gz /data

# Restore MLflow data
docker run --rm -v aardvark_mlflow-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/mlflow-backup.tar.gz -C /
```

### Database
PostgreSQL data is stored in Docker volume `db-data`:
```bash
# Backup database
docker compose exec db pg_dump -U postgres aardvark_db > backup.sql

# Restore database
cat backup.sql | docker compose exec -T db psql -U postgres aardvark_db
```

## Advanced Configuration

### Change UMAP Parameters
Edit `app/celery_workers/agents.py`:
```python
# In DKLAgent.fit()
umap_model = UMAP(
    n_components=3,      # Change embedding dimensions
    n_neighbors=15,      # Adjust neighborhood size
    min_dist=0.1         # Adjust minimum distance
)
```

### Adjust GP Training
```python
# In DKLAgent.fit()
training_iterations = 100  # Increase for better fit
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Adjust learning rate
```

### Modify Acquisition Function
```python
# In DKLAgent.ask()
# Current: uncertainty × intensity
acquisition = stds * np.clip(intensity_mean, 0, 1)

# Pure uncertainty
acquisition = stds.sum(axis=1)

# Upper confidence bound
beta = 2.0
acquisition = means + beta * stds
```

## Stopping the System

```bash
# Stop all services
docker compose down

# Stop and remove all data (WARNING: deletes all experiments)
docker compose down -v
```

## Getting Help

If you encounter issues:

1. Check logs: `docker compose logs -f [service_name]`
2. Verify all services are running: `docker compose ps`
3. Check MLflow UI for error messages
4. Review dashboard for data flow issues

## Next Steps

- Explore MLflow UI to understand experiment tracking
- Customize the acquisition function for your use case
- Integrate with your real experimental hardware
- Experiment with different dimensionality reduction techniques
- Tune GP hyperparameters for better predictions
