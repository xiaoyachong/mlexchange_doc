# ============================================
# YOUR EXISTING NOTEBOOK CODE (keep as-is)
# ============================================

# ... your imports ...
# ... your toroid() function ...
# ... your digestion() function ...
# ... your dofs and objectives definition ...
# ... your agent creation ...


# ============================================
# STEP 1: Run BLOP to find optimal x_rot, y_rot
# ============================================

print("Starting optimization to find best x_rot and y_rot...")
print("="*60)

# Initial random exploration
print("Phase 1: Random sampling (32 points)...")
RE(agent.learn("quasi-random", n=32))
best_so_far = agent.table["beamsize"].min()
print(f"Best beamsize after random: {best_so_far:.6f} μm")

# Intelligent Bayesian optimization
print("\nPhase 2: Bayesian optimization (8 iterations x 4 points)...")
RE(agent.learn("qei", n=4, iterations=8))

# Show the result
best = agent.best
print("\n" + "="*60)
print("✓ OPTIMIZATION COMPLETE")
print("="*60)
print(f"BLOP found optimal x_rot: {best['x_rot']:.6f}")
print(f"BLOP found optimal y_rot: {best['y_rot']:.6f}")
print(f"Minimum beamsize:          {best['beamsize']:.6f} μm")
print(f"Total samples evaluated:   {len(agent.table)}")
print("="*60)

# Visualize what BLOP found
agent.plot_objectives()


# ============================================
# STEP 2: Log to MLflow
# ============================================

import mlflow
import mlflow.pyfunc
import pickle

print("\nLogging results to MLflow...")

mlflow.set_experiment("shadow_toroid_optimization")

with mlflow.start_run():
    
    # Save the optimal parameters BLOP found
    optimal_params = {
        'x_rot': float(best['x_rot']),
        'y_rot': float(best['y_rot']),
        'beamsize': float(best['beamsize'])
    }
    
    # Log as metrics
    mlflow.log_metric("optimal_x_rot", optimal_params['x_rot'])
    mlflow.log_metric("optimal_y_rot", optimal_params['y_rot'])
    mlflow.log_metric("optimal_beamsize", optimal_params['beamsize'])
    
    # Save optimization history
    agent.table.to_csv("history.csv", index=False)
    mlflow.log_artifact("history.csv")
    
    # Save plots
    fig = agent.plot_objectives()
    mlflow.log_figure(fig, "objectives.png")
    
    # Log the model (with wrapper)
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save optimal params to file
        params_file = os.path.join(tmpdir, "optimal_params.pkl")
        with open(params_file, "wb") as f:
            pickle.dump(optimal_params, f)
        
        # Define the model wrapper
        class ShadowSim(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                with open(context.artifacts["optimal_params"], "rb") as f:
                    self.optimal_params = pickle.load(f)
            
            def predict(self, context, model_input):
                # Empty input = use optimal params
                if len(model_input) == 0:
                    x = self.optimal_params['x_rot']
                    y = self.optimal_params['y_rot']
                    return pd.DataFrame([{
                        'x_rot': x, 'y_rot': y, 
                        'beamsize': toroid(x, y),
                        'note': 'optimal'
                    }])
                # Custom input = use those params
                results = []
                for _, row in model_input.iterrows():
                    results.append({
                        'x_rot': row['x_rot'],
                        'y_rot': row['y_rot'],
                        'beamsize': toroid(row['x_rot'], row['y_rot'])
                    })
                return pd.DataFrame(results)
        
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="simulator",
            python_model=ShadowSim(),
            artifacts={"optimal_params": params_file}
        )
    
    run_id = mlflow.active_run().info.run_id
    
    print("\n✓ Logged to MLflow!")
    print(f"Run ID: {run_id}")
    print(f"Model: runs:/{run_id}/simulator")


# ============================================
# STEP 3: Load and use the model later
# ============================================

print("\n" + "="*60)
print("Loading the saved model...")
print("="*60)

# Load the model
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/simulator")

# Test 1: Simulate at optimal parameters (empty input)
print("\n1. Simulation at optimal parameters:")
result = model.predict(pd.DataFrame())
print(result)

# Test 2: Simulate at custom parameters
print("\n2. Simulation at custom parameters:")
test_points = pd.DataFrame({
    'x_rot': [0.0, 0.05, -0.05],
    'y_rot': [0.0, 0.03, -0.03]
})
results = model.predict(test_points)
print(results)

print("\n✓ Complete! The model remembers the optimal x_rot and y_rot that BLOP found.")
print(f"  Save this run_id: {run_id}")
print(f"  Load anytime with: mlflow.pyfunc.load_model('runs:/{run_id}/simulator')")
