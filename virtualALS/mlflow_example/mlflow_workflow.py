"""
Complete workflow: Use BLOP to find optimal x_rot, y_rot, then log with MLflow
"""

import numpy as np
import pandas as pd
import Shadow
from blop import DOF, Objective, Agent
import mlflow
import mlflow.pyfunc
import pickle


# ==============================================================================
# Step 1: Define the Shadow simulation
# ==============================================================================

def toroid(x_rot, y_rot):
    """Shadow simulation - returns beamsize for given x_rot, y_rot."""
    beam = Shadow.Beam()
    oe0 = Shadow.Source()
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()

    oe1.X_ROT = x_rot
    oe1.Y_ROT = y_rot

    oe0.FDISTR = 3
    oe0.F_PHOT = 0
    oe0.HDIV1 = 0.0
    oe0.HDIV2 = 0.0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.ISTAR1 = 5676561
    oe0.PH1 = 1000.0
    oe0.SIGDIX = 5e-05
    oe0.SIGDIZ = 5e-05
    oe0.SIGMAX = 1e-05
    oe0.SIGMAZ = 1e-05
    oe0.VDIV1 = 0.0
    oe0.VDIV2 = 0.0
    
    oe1.DUMMY = 100.0
    oe1.FMIRR = 3
    oe1.FWRITE = 1
    oe1.F_EXT = 1
    oe1.F_MOVE = 1
    oe1.R_MAJ = 305.3065
    oe1.R_MIN = 0.655
    oe1.T_IMAGE = 0.0

    oe2.ALPHA = 90.0
    oe2.DUMMY = 100.0
    oe2.FCYL = 1
    oe2.FMIRR = 2
    oe2.FWRITE = 1
    oe2.F_DEFAULT = 0
    oe2.SIMAG = 5.0
    oe2.SSOUR = 1000000.0
    oe2.THETA = 88.0
    oe2.T_IMAGE = 5.0
    oe2.T_SOURCE = 5.0

    beam.genSource(oe0)
    beam.traceOE(oe1, 1)
    beam.traceOE(oe2, 2)

    x_fhwm_m = beam.histo1(1)['fwhm']
    y_fhwm_m = beam.histo1(3)['fwhm']

    a = x_fhwm_m * 1e6
    b = y_fhwm_m * 1e6
    size_um = np.sqrt(a**2 + b**2)
   
    return size_um


# ==============================================================================
# Step 2: Setup BLOP to find optimal x_rot, y_rot
# ==============================================================================

def digestion(db, uid):
    """Tell BLOP how to calculate beamsize from the data."""
    products = db[uid].table()
    
    for index, entry in products.iterrows():
        products.loc[index, "beamsize"] = toroid(entry.x_rot, entry.y_rot)
    
    return products


def run_blop_optimization(db):
    """
    Use BLOP to find the best x_rot and y_rot that minimize beamsize.
    
    Returns:
    --------
    agent : blop.Agent
        The trained agent with all sampled points and the best parameters
    """
    
    # Define the degrees of freedom (parameters to optimize)
    dofs = [
        DOF(name="x_rot", search_bounds=(-0.2, 0.2)),
        DOF(name="y_rot", search_bounds=(-0.2, 0.2)),
    ]
    
    # Define the objective (what we want to minimize)
    objectives = [
        Objective(name="beamsize", description="beam size", target="min")
    ]
    
    # Create the BLOP agent
    agent = Agent(
        dofs=dofs,
        objectives=objectives,
        digestion=digestion,
        db=db,
    )
    
    print("Starting BLOP optimization...")
    print("="*60)
    
    # Step 1: Initial random sampling to explore the space
    print("\n1. Initial random sampling (32 points)...")
    from bluesky import RunEngine
    RE = RunEngine()
    RE(agent.learn("quasi-random", n=32))
    
    # Check progress
    best_so_far = agent.table["beamsize"].min()
    print(f"   Best beamsize after random sampling: {best_so_far:.6f} μm")
    
    # Step 2: Intelligent optimization using Bayesian optimization
    print("\n2. Bayesian optimization (8 iterations x 4 points)...")
    RE(agent.learn("qei", n=4, iterations=8))
    
    # Final result
    best = agent.best
    print("\n" + "="*60)
    print("✓ OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best x_rot found: {best['x_rot']:.6f}")
    print(f"Best y_rot found: {best['y_rot']:.6f}")
    print(f"Minimum beamsize:  {best['beamsize']:.6f} μm")
    print(f"Total samples evaluated: {len(agent.table)}")
    print("="*60 + "\n")
    
    return agent


# ==============================================================================
# Step 3: MLflow Model Wrapper
# ==============================================================================

class ShadowSimulator(mlflow.pyfunc.PythonModel):
    """Wrapper to save/load the Shadow simulator with optimal parameters."""
    
    def __init__(self, optimal_params=None):
        self.optimal_params = optimal_params
    
    def load_context(self, context):
        with open(context.artifacts["optimal_params"], "rb") as f:
            self.optimal_params = pickle.load(f)
    
    def predict(self, context, model_input):
        """Run Shadow simulation."""
        # If empty input, use optimal parameters
        if model_input is None or len(model_input) == 0:
            x_rot = self.optimal_params['x_rot']
            y_rot = self.optimal_params['y_rot']
            beamsize = toroid(x_rot, y_rot)
            
            return pd.DataFrame([{
                'x_rot': x_rot,
                'y_rot': y_rot,
                'beamsize': beamsize,
                'note': 'optimal_parameters'
            }])
        
        # Run simulation for provided points
        results = []
        for _, row in model_input.iterrows():
            x_rot = row['x_rot']
            y_rot = row['y_rot']
            beamsize = toroid(x_rot, y_rot)
            results.append({'x_rot': x_rot, 'y_rot': y_rot, 'beamsize': beamsize})
        
        return pd.DataFrame(results)


# ==============================================================================
# Step 4: Log to MLflow
# ==============================================================================

def log_optimization_to_mlflow(agent, experiment_name="shadow_optimization"):
    """
    Log the BLOP optimization results and model to MLflow.
    
    This saves:
    - The optimal x_rot, y_rot found by BLOP
    - All sampled points (optimization history)
    - A loadable model for running new simulations
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        
        # Get the best parameters found by BLOP
        best = agent.best
        optimal_params = {
            'x_rot': float(best['x_rot']),
            'y_rot': float(best['y_rot']),
            'beamsize': float(best['beamsize'])
        }
        
        # Log the optimal parameters as metrics
        mlflow.log_metric("optimal_x_rot", optimal_params['x_rot'])
        mlflow.log_metric("optimal_y_rot", optimal_params['y_rot'])
        mlflow.log_metric("optimal_beamsize", optimal_params['beamsize'])
        mlflow.log_metric("n_samples", len(agent.table))
        
        # Log optimization history
        agent.table.to_csv("optimization_history.csv", index=False)
        mlflow.log_artifact("optimization_history.csv")
        
        # Log plots
        try:
            fig = agent.plot_objectives()
            mlflow.log_figure(fig, "objective_landscape.png")
        except:
            pass
        
        # Save and log the model
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = os.path.join(tmpdir, "optimal_params.pkl")
            with open(params_path, "wb") as f:
                pickle.dump(optimal_params, f)
            
            mlflow.pyfunc.log_model(
                artifact_path="shadow_simulator",
                python_model=ShadowSimulator(optimal_params=optimal_params),
                artifacts={"optimal_params": params_path},
                pip_requirements=["numpy", "pandas", "oasys-shadow3"]
            )
        
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "="*60)
        print("✓ Results logged to MLflow")
        print("="*60)
        print(f"Run ID: {run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Model URI: runs:/{run_id}/shadow_simulator")
        print("="*60 + "\n")
        
        return run_id


# ==============================================================================
# Step 5: Load and use the saved model
# ==============================================================================

def use_saved_model(run_id):
    """Load the model from MLflow and run simulations."""
    
    print(f"Loading model from run: {run_id}")
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/shadow_simulator")
    
    # Simulate at optimal parameters
    print("\n1. Simulating at optimal parameters...")
    result_optimal = model.predict(pd.DataFrame())
    print(result_optimal)
    
    # Simulate at custom parameters
    print("\n2. Simulating at custom parameters...")
    custom_points = pd.DataFrame({
        'x_rot': [0.0, 0.05, -0.05],
        'y_rot': [0.0, 0.02, -0.02]
    })
    result_custom = model.predict(custom_points)
    print(result_custom)
    
    return model


# ==============================================================================
# COMPLETE EXAMPLE - Put this in your notebook
# ==============================================================================

if __name__ == "__main__":
    
    # Assuming you have 'db' from your notebook setup:
    # from blop.utils import prepare_re_env
    # %run -i $prepare_re_env.__file__ --db-type=temp
    
    # FULL WORKFLOW:
    
    # 1. Run BLOP to find optimal x_rot, y_rot
    print("STEP 1: Running BLOP optimization...")
    agent = run_blop_optimization(db)
    
    # 2. Log everything to MLflow
    print("\nSTEP 2: Logging to MLflow...")
    run_id = log_optimization_to_mlflow(agent, experiment_name="shadow_toroid")
    
    # 3. Later, load and use the model
    print("\nSTEP 3: Loading and using the saved model...")
    model = use_saved_model(run_id)
    
    print("\n✓ Done! You can now:")
    print(f"  - View results in MLflow UI: mlflow ui")
    print(f"  - Load model anytime with: mlflow.pyfunc.load_model('runs:/{run_id}/shadow_simulator')")
