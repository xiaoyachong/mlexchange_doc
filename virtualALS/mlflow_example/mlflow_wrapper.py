import mlflow
import mlflow.pyfunc
import pickle
import numpy as np
import Shadow
from blop import Agent, DOF, Objective


class ShadowBLOPModel(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for Shadow simulation with BLOP optimization agent.
    """
    
    def __init__(self, agent=None):
        self.agent = agent
        
    def load_context(self, context):
        """Load the agent from artifacts."""
        import pickle
        with open(context.artifacts["agent_state"], "rb") as f:
            self.agent = pickle.load(f)
    
    def predict(self, context, model_input):
        """
        Make predictions using the trained agent.
        
        Parameters:
        -----------
        model_input : pd.DataFrame
            DataFrame with columns 'x_rot' and 'y_rot'
            
        Returns:
        --------
        pd.DataFrame with predicted beamsize
        """
        predictions = []
        for _, row in model_input.iterrows():
            # Use the agent's model to predict (not the actual Shadow simulation)
            x_rot = row['x_rot']
            y_rot = row['y_rot']
            
            # Get prediction from the trained Gaussian Process model
            # This avoids running expensive Shadow simulations during inference
            pred_mean, pred_std = self.agent.predict([[x_rot, y_rot]])
            
            predictions.append({
                'x_rot': x_rot,
                'y_rot': y_rot,
                'predicted_beamsize': pred_mean[0],
                'uncertainty': pred_std[0]
            })
        
        import pandas as pd
        return pd.DataFrame(predictions)


def toroid(X_ROT, Y_ROT):
    """Shadow simulation function."""
    beam = Shadow.Beam()
    oe0 = Shadow.Source()
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()

    oe1.X_ROT = X_ROT
    oe1.Y_ROT = Y_ROT

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


def log_shadow_blop_model(agent, experiment_name="shadow_beamline_optimization"):
    """
    Log the BLOP agent and metadata to MLflow.
    
    Parameters:
    -----------
    agent : blop.Agent
        Trained BLOP agent
    experiment_name : str
        Name of the MLflow experiment
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        
        # Log parameters
        mlflow.log_param("n_initial_samples", len(agent.table))
        mlflow.log_param("acquisition_function", "qei")
        mlflow.log_param("dof_names", [dof.name for dof in agent.dofs])
        mlflow.log_param("objective_names", [obj.name for obj in agent.objectives])
        
        # Log the search bounds
        for dof in agent.dofs:
            mlflow.log_param(f"{dof.name}_bounds", dof.search_bounds)
        
        # Log metrics from the best point
        best = agent.best
        mlflow.log_metric("best_beamsize", best["beamsize"])
        mlflow.log_metric("best_x_rot", best["x_rot"])
        mlflow.log_metric("best_y_rot", best["y_rot"])
        
        # Log the optimization history as metrics over iterations
        table = agent.table
        for idx, row in table.iterrows():
            mlflow.log_metric("beamsize", row["beamsize"], step=idx)
        
        # Save agent state to a temporary file
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = os.path.join(tmpdir, "agent_state.pkl")
            
            # Save the agent
            with open(agent_path, "wb") as f:
                pickle.dump(agent, f)
            
            # Create artifacts dictionary
            artifacts = {"agent_state": agent_path}
            
            # Log the model with pyfunc
            mlflow.pyfunc.log_model(
                artifact_path="shadow_blop_model",
                python_model=ShadowBLOPModel(agent),
                artifacts=artifacts,
                conda_env={
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={mlflow.pyfunc.get_default_pip_requirements()[0].split('==')[1]}",
                        "pip",
                        {
                            "pip": [
                                "mlflow",
                                "numpy",
                                "pandas",
                                "scikit-learn",
                                "blop",  # Make sure this is available
                                "Shadow",  # Shadow library
                                "botorch",
                                "gpytorch",
                            ]
                        }
                    ],
                    "name": "shadow_blop_env"
                }
            )
        
        # Log visualizations
        fig = agent.plot_objectives()
        mlflow.log_figure(fig, "objective_landscape.png")
        
        # Log acquisition function plot
        fig_acq = agent.plot_acquisition(acq_func="qei")
        mlflow.log_figure(fig_acq, "acquisition_function.png")
        
        # Log the training data as an artifact
        table.to_csv("training_data.csv", index=False)
        mlflow.log_artifact("training_data.csv")
        os.remove("training_data.csv")
        
        print(f"Model logged to MLflow run: {run.info.run_id}")
        print(f"Model URI: runs:/{run.info.run_id}/shadow_blop_model")
        
        return run.info.run_id


# Example usage after training your agent:
# Assuming you've already trained your agent as in your notebook
# run_id = log_shadow_blop_model(agent)

# To load and use the model later:
def load_and_predict(run_id):
    """Load model and make predictions."""
    model_uri = f"runs:/{run_id}/shadow_blop_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Create test data
    import pandas as pd
    test_data = pd.DataFrame({
        'x_rot': [0.0, 0.1, -0.1],
        'y_rot': [0.0, 0.05, -0.05]
    })
    
    predictions = loaded_model.predict(test_data)
    return predictions
