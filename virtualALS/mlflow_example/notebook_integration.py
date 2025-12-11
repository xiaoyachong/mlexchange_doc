# Add this cell after your agent training is complete
# (After the RE(agent.learn(...)) calls)

import mlflow
from mlflow_shadow_wrapper import log_shadow_blop_model, load_and_predict

# Set your MLflow tracking URI (optional - defaults to local ./mlruns)
# mlflow.set_tracking_uri("http://your-mlflow-server:5000")

# Log the trained agent to MLflow
run_id = log_shadow_blop_model(
    agent=agent,
    experiment_name="shadow_toroid_optimization"
)

print(f"✓ Model logged successfully!")
print(f"Run ID: {run_id}")
print(f"Best beamsize found: {agent.best['beamsize']:.4f}")
print(f"At position: x_rot={agent.best['x_rot']:.4f}, y_rot={agent.best['y_rot']:.4f}")

# Later, to load and use the model:
# loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/shadow_blop_model")
# predictions = loaded_model.predict(test_data)


# Alternative: Log during training with auto-logging
# You can also log metrics during the optimization loop:
import mlflow

mlflow.set_experiment("shadow_toroid_optimization_live")

with mlflow.start_run():
    # Log initial parameters
    mlflow.log_params({
        "initial_samples": 32,
        "learning_iterations": 8,
        "points_per_iteration": 4,
        "acquisition_function": "qei"
    })
    
    # Initial sampling
    RE(agent.learn("quasi-random", n=32))
    mlflow.log_metric("samples_collected", 32, step=0)
    
    # Iterative learning with logging
    for iteration in range(8):
        RE(agent.learn("qei", n=4, iterations=1))
        
        # Log metrics after each iteration
        best = agent.best
        mlflow.log_metric("best_beamsize", best["beamsize"], step=iteration+1)
        mlflow.log_metric("samples_collected", len(agent.table), step=iteration+1)
        
        # Log plots periodically
        if (iteration + 1) % 2 == 0:
            fig = agent.plot_objectives()
            mlflow.log_figure(fig, f"objectives_iter_{iteration+1}.png")
    
    # Final model logging
    run_id = mlflow.active_run().info.run_id
    log_shadow_blop_model(agent, experiment_name="shadow_toroid_optimization_live")
