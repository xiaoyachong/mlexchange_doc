import numpy as np
from scipy.interpolate import griddata
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, basinhopping

from . import tasks
from .celery_app import app
import matplotlib.pyplot as plt

from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

import time
import pickle
import mlflow
import mlflow.pytorch
import json

# This is an adaptation of the Agent base class from bluesky-adaptive that works for my use

class Agent(ABC):
    @abstractmethod
    def tell(self, x, y) -> Dict[str, ArrayLike]:
        """Tell the agent about some new data"""
        ...

    @abstractmethod
    def tell_many(self, x, y) -> Dict[str, ArrayLike]:
        """Tell the agent about some new data"""
        ...

    @abstractmethod
    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        """Ask the agent for a new batch of points to measure."""
        ...

    def report(self, **kwargs) -> Dict[str, ArrayLike]:
        """Create a report given the data observed by the agent."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Short string name"""
        return "agent"


class IntensityAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def tell(self, x, y):
        return x, np.sum(y, axis=1)
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    

from sklearn.decomposition import KernelPCA
class KPCAAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, **kwargs):
        print("Creating PCAAgent")
        self.kwargs = kwargs
        if kwargs.get('n_components') is None:
            raise ValueError("Please enter a non-None value for n_components of PCA.")
        kwargs['kernel'] = 'rbf' if kwargs.get('kernel') is None else kwargs.get('kernel')

    def tell(self, x, y):
        print(f"PCAAgent told about new data: {x.shape=}, {y.shape=}")
        if len(y) < self.kwargs.get('n_components'):
            self.PCA = KernelPCA(n_components=None, **{k:v for k,v in self.kwargs.items() if k!='n_components'})
            return x, self.PCA.fit_transform(y)
        
        self.PCA = KernelPCA(**self.kwargs)
        return x, self.PCA.fit_transform(y)
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    

from cuml import IncrementalPCA as PCA
class PCAAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, **kwargs):
        print("Creating PCAAgent")
        self.kwargs = kwargs
        self.n_components = kwargs.get('n_components') if kwargs.get('n_components') is not None else 50

    def tell(self, x, y):
        print(f"PCAAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        
        self.PCA = PCA(n_components=min(self.n_components, len(y)))
        result = self.PCA.fit_transform(y.astype(np.float32))
        end = time.perf_counter_ns()
        print(f"PCA took {(end-start)/1e6:.02f}ms to fit.")
        return x, result
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError

from cuml import UMAP
class UMAPAgent(Agent):
    def __init__(self, **kwargs):
        print("Creating UMAPAgent")
        self.kwargs = kwargs
        self.n_components = kwargs.get('n_components') if kwargs.get('n_components') is not None else 2

    def tell(self, x, y):
        print(f"UMAPAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        self.UMAP = UMAP(n_components=self.n_components)
        new_y = self.UMAP.fit_transform(y.astype(np.float32))
        end = time.perf_counter_ns()
        print(f"UMAP took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError

from cuml.cluster.hdbscan import HDBSCAN
class HDBSCANAgent(Agent):
    def __init__(self, *args, **kwargs):
        print("Creating HDBSCANAgent")
        self.args = args
        self.kwargs = kwargs

    def tell(self, x, y):
        print(f"HDBSCANAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        if len(y) < 2:
            self.labels = np.zeros(len(y))
            new_y = np.zeros(len(y))
        else:
            self.HDBSCAN = HDBSCAN(**self.kwargs)
            self.labels = self.HDBSCAN.fit(y.astype(np.float32))
            new_y = self.HDBSCAN.labels_ + 1
        end = time.perf_counter_ns()
        print(f"HDBSCAN took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError


from cuml.cluster import KMeans
class KMeansAgent(Agent):
    """A simple naive agent that cycles samples sequentially in environment space"""

    def __init__(self, *args, **kwargs):
        print("Creating KMeansAgent")
        self.args = args
        self.kwargs = kwargs
        self.n_clusters = kwargs['n_clusters']
        self.experiment_id = kwargs.get('experiment_id')
        self.KMeans = None

    def tell(self, x, y):
        print(f"KMeansAgent told about new data: {x.shape=}, {y.shape=}")
        start = time.perf_counter_ns()
        if len(y) < self.n_clusters:
            self.labels = np.zeros(len(y))
            new_y = np.zeros(len(y))
        else:
            self.KMeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.KMeans.fit(y.astype(np.float32))
            self.labels = self.KMeans.labels_
            new_y = self.labels
        self.x = x
        self.y = new_y
        end = time.perf_counter_ns()
        print(f"KMeans took {(end-start)/1e6:.02f}ms to fit.")
        return x, new_y
    
    def ask(self, batch_size):
        raise NotImplementedError
    
    def tell_many(self, x, y):
        raise NotImplementedError
    
    def report(self, meshgrid):
        print("Reporting KMeansAgent")
        if self.KMeans is None:
            print("No KMeans model to report")
            return
        
        label_grid = griddata(self.x, self.y, (meshgrid[0], meshgrid[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=label_grid, name='kmeans_labels',
                            extra_data = dict(n_measured=len(self.y)),
                            )


class DKLModel(gpytorch.models.ExactGP):
        
    def __init__(self, train_x, train_y, likelihood):
        super(DKLModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=train_y.shape[-1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=train_y.shape[-1], rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class DKLAgent(Agent):
    def __init__(self, input_bounds, input_min_spacings, experiment_id, data_ids):
        print("Creating Classification GP Agent")
        self.intensity_factor = 1
        self.umap_factor = 1
        self.data_ids = data_ids
        self.input_bounds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("THE DEVICE BEING USED IS: ", self.device)
        p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
        self.meshgrid_points = np.meshgrid(*p)
        self.plotting_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
        p = [np.linspace(low, high, 4) for (low, high) in self.input_bounds]
        self.scipy_meshgrid = np.meshgrid(*p)
        self.optimize_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.scipy_meshgrid],dim=1)
        self.experiment_id = experiment_id
        
        # MLflow setup
        from db.config import settings
        self.mlflow_experiment_name = f"AARDVARK_Experiment_{experiment_id}"
        self.iteration = 0
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        try:
            mlflow.create_experiment(self.mlflow_experiment_name)
        except:
            pass  # Experiment already exists
        
        mlflow.set_experiment(self.mlflow_experiment_name)
        print(f"MLflow tracking URI set to: {settings.MLFLOW_TRACKING_URI}")
        print(f"MLflow experiment: {self.mlflow_experiment_name}")

    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        if True:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.float32))
            self.intensities = self.targets.sum(axis=1)
            self.intensities = self.intensities / self.intensities.max()
        
        start = time.perf_counter_ns()
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"iteration_{self.iteration}"):
            # Log parameters
            mlflow.log_params({
                "n_measurements": len(x),
                "n_features": y.shape[1] if len(y.shape) > 1 else 1,
                "input_bounds_x": str(self.input_bounds[0]),
                "input_bounds_y": str(self.input_bounds[1]),
                "device": str(self.device),
                "iteration": self.iteration,
                "intensity_factor": self.intensity_factor,
                "umap_factor": self.umap_factor
            })
            
            # Fit model
            self.fit()
            
            # Log training time
            training_time = (time.perf_counter_ns() - start) / 1e6
            mlflow.log_metric("training_time_ms", training_time)
            mlflow.log_metric("total_data_points", len(self.inputs))
            
            self.iteration += 1
        
        print(f"GP took {training_time:.02f}ms to fit.")
        return x, y

    def fit(self):
        train_x = self.inputs
        train_y = self.targets
        training_iterations = 100

        # ===== UMAP Training =====
        print("Fitting UMAP")
        start_umap = time.perf_counter()
        umap_model = UMAP(n_components=3)
        encoded_y = umap_model.fit(train_y).transform(train_y)
        encoded_y = torch.as_tensor(encoded_y, device=self.device)
        umap_time = time.perf_counter() - start_umap
        
        print(f"UMAP fitted: {type(encoded_y)}")
        
        # Log UMAP metrics
        mlflow.log_metric("umap_fit_time_sec", umap_time)
        mlflow.log_param("umap_n_components", 3)
        mlflow.log_metric("umap_embedding_variance", float(encoded_y.var().cpu()))
        
        # Save UMAP coordinates
        tasks.save_report(experiment_id=self.experiment_id, name='umap_coords', 
                          data={
                                'umap_coords': encoded_y.cpu().numpy().tolist(),
                                'xy_coords': self.inputs.cpu().numpy().tolist(),
                                'n_measured': len(self.inputs),
                                'data_ids': self.data_ids,
                            })
    
        # ===== GP Training =====
        print("Fitting GP")
        gp_targets = torch.cat([encoded_y, self.intensities.unsqueeze(-1)], dim=-1)
        print(f"{encoded_y.shape=}, {self.intensities.shape=}, {gp_targets.shape=}")
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=encoded_y.shape[-1] + 1)
        
        model = DKLModel(train_x, gp_targets, likelihood)
        model.to(self.device)
        likelihood.to(self.device)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        print("Training GP")
        start_gp = time.perf_counter()
        losses = []
        
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, gp_targets)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Log every 10 iterations
            if i % 10 == 0:
                mlflow.log_metric("gp_loss", loss.item(), step=i)

        gp_time = time.perf_counter() - start_gp
        
        # Log GP metrics
        mlflow.log_metric("gp_fit_time_sec", gp_time)
        mlflow.log_metric("gp_final_loss", losses[-1])
        mlflow.log_metric("gp_training_iterations", training_iterations)
        mlflow.log_param("gp_learning_rate", 0.1)
        mlflow.log_param("gp_kernel_type", "RBFKernel")
        mlflow.log_param("gp_num_tasks", 4)
        
        # Set to eval mode
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.model = model
            self.likelihood = likelihood
            
            # Save PyTorch model
            try:
                mlflow.pytorch.log_model(
                    model, 
                    "gp_model",
                    code_paths=["app/celery_workers/agents.py"]
                )
                
                # Save likelihood separately
                likelihood_path = f"/tmp/likelihood_state_{self.experiment_id}_{self.iteration}.pth"
                torch.save(likelihood.state_dict(), likelihood_path)
                mlflow.log_artifact(likelihood_path, "model")
                
                print("Model logged to MLflow successfully")
            except Exception as e:
                print(f"Error logging model to MLflow: {e}")

    def ask(self, batch_size):
        print("Sampling...")
        
        # Start nested run for predictions
        with mlflow.start_run(run_name=f"ask_iteration_{self.iteration-1}", nested=True):
            max_n_samples = 10_000
            if len(self.plotting_positions) >= max_n_samples:
                test_x = self.plotting_positions[np.random.choice(len(self.plotting_positions), max_n_samples)]
            else:
                test_x = self.plotting_positions
            
            # Log prediction parameters
            mlflow.log_params({
                "batch_size": batch_size,
                "n_candidate_points": len(test_x),
                "intensity_factor": self.intensity_factor,
                "umap_factor": self.umap_factor
            })
            
            print(f"Evaluating model on {len(test_x)} positons...")
            start_pred = time.perf_counter()
            
            evaluation = self.likelihood(self.model(test_x))
            test_x = test_x.detach().cpu().numpy()
            
            means = evaluation.mean.detach().cpu().numpy()
            intensity_mean = means[:,-1]
            means = means[:,:-1]
            print(f'{means.shape=}')
            
            stds = evaluation.stddev
            intensity_std = stds[:,-1].detach().cpu().numpy()
            stds -= stds.min(axis=0, keepdim=True).values
            stds /= stds.max(axis=0, keepdim=True).values
            stds = stds.sum(axis=1).detach().cpu().numpy() / 4

            stds = stds * np.clip(intensity_mean, 0, 1)
            
            pred_time = time.perf_counter() - start_pred
            
            # Log prediction metrics
            mlflow.log_metric("prediction_time_sec", pred_time)
            mlflow.log_metric("mean_uncertainty", float(stds.mean()))
            mlflow.log_metric("max_uncertainty", float(stds.max()))
            mlflow.log_metric("min_uncertainty", float(stds.min()))
            mlflow.log_metric("std_uncertainty", float(stds.std()))
            mlflow.log_metric("mean_predicted_intensity", float(intensity_mean.mean()))
            mlflow.log_metric("max_predicted_intensity", float(intensity_mean.max()))
            
            xi = (self.meshgrid_points[0], self.meshgrid_points[1])
            method = 'nearest'
            
            # Generate and log visualizations
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'Acquisition function',
                                test_x,
                                stds,
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'GP Intensities',
                                test_x,
                                intensity_mean,
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'GP Intensity Uncertainty',
                                test_x,
                                intensity_std,
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'UMAP x',
                                test_x,
                                means[:,0],
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'UMAP y',
                                test_x,
                                means[:,1],
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'UMAP z',
                                test_x,
                                means[:,2],
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'UMAP interpolated coords',
                                test_x,
                                means,
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            scaled_means = (255 * (means - means.min(axis=0)[np.newaxis, ...]) / (means.max(axis=0)[np.newaxis, ...] - means.min(axis=0)[np.newaxis, ...])).astype(np.uint8)
            
            app.send_task("celery_workers.tasks.image_report", args=(
                                self.experiment_id,
                                'GP est. UMAP as RGB',
                                test_x,
                                scaled_means,
                                xi,
                                method,
                                dict(n_measured=len(self.inputs)),
                                ))
            
            high_stds = stds
            choice_indices = np.argsort(high_stds.flatten())

            num_most_uncertain_to_measure_first = 1
            most_uncertain_indices = choice_indices[-num_most_uncertain_to_measure_first:][::-1]
            p = high_stds[:-num_most_uncertain_to_measure_first]
            if p.sum()!=0:
                p /= p.sum()
                probabalistic_indices = np.random.choice(choice_indices[:-num_most_uncertain_to_measure_first], 
                                                    size=batch_size-num_most_uncertain_to_measure_first,
                                                    replace=False,
                                                    p=p)
                selected_indices =  np.concatenate([most_uncertain_indices, probabalistic_indices])
            else:
                selected_indices = choice_indices[-batch_size:][::-1]
            
            next_positions = test_x[selected_indices]
            
            # Log selected positions
            mlflow.log_param("n_positions_suggested", len(next_positions))
            mlflow.log_metric("top_acquisition_value", float(stds[selected_indices[0]]))
            mlflow.log_metric("mean_selected_acquisition", float(stds[selected_indices].mean()))
            
            # Save positions as artifact
            positions_dict = {
                "positions": next_positions.tolist(),
                "acquisition_values": stds[selected_indices].tolist(),
                "predicted_intensities": intensity_mean[selected_indices].tolist()
            }
            positions_path = f"/tmp/suggested_positions_{self.experiment_id}_{self.iteration}.json"
            with open(positions_path, "w") as f:
                json.dump(positions_dict, f, indent=2)
            mlflow.log_artifact(positions_path, "suggestions")
            
            return next_positions

    def tell_many(self, x, y):
        raise NotImplementedError


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPytorchAgent(Agent):
    def __init__(self, input_bounds, input_min_spacings, experiment_id):
        print("Creating Classification GP Agent")
        self.input_bounds = input_bounds
        self.input_min_spacings = input_min_spacings
        self.inputs=None
        self.targets=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("THE DEVICE BEING USED IS: ", self.device)
        p = [np.arange(low, high, delta) for (low, high), delta in zip(self.input_bounds, self.input_min_spacings)]
        self.meshgrid_points = np.meshgrid(*p)
        self.plotting_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.meshgrid_points],dim=1)
        p = [np.linspace(low, high, 4) for (low, high) in self.input_bounds]
        self.scipy_meshgrid = np.meshgrid(*p)
        self.optimize_positions = torch.stack([torch.tensor(arr.flatten(),dtype=torch.float32).to(self.device) for arr in self.scipy_meshgrid],dim=1)
        self.experiment_id = experiment_id

    def tell(self, x, y):
        print(f"GPytorchAgent told about new data: {x.shape=}, {y.shape=}")
        if True:
            self.inputs = torch.atleast_2d(torch.tensor(x, device=self.device, dtype=torch.float32))
            self.targets = torch.atleast_1d(torch.tensor(y, device=self.device, dtype=torch.int32))
        start = time.perf_counter_ns()
        self.fit()
        end = time.perf_counter_ns()
        print(f"GP took {(end-start)/1e6:.02f}ms to fit.")
        return x, y

    def fit(self):
        print("Fitting GP")
        train_x = self.inputs
        train_y = self.targets

        likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=False)
        model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)

        model.to(self.device)
        likelihood.to(self.device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        training_iterations = 50
        print("Training GP")
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, likelihood.transformed_targets).sum()
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.model = model
            self.likelihood = likelihood

    def ask(self, batch_size):
        max_n_samples = 10_000
        if len(self.plotting_positions) > max_n_samples:
            test_x = self.plotting_positions[np.random.choice(len(self.plotting_positions), max_n_samples)]
        else:
            test_x = self.plotting_positions
        
        print(f"Evaluating model on {len(test_x)} positons...")
        evaluation = self.model(test_x)
        test_x = test_x.detach().cpu().numpy()
        
        print("Sampling...")
        pred_samples = evaluation.sample(torch.Size((256,))).exp()
        info = (pred_samples / pred_samples.sum(-2, keepdim=True))
        stds = info.std(0)
        stds = stds.sum(0).detach().cpu().numpy()
        
        partial = tasks.plot_griddata.s(xs=test_x, ys=np.copy(stds))
        grid_stds = griddata(test_x, stds, (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id, 
                            matrix=grid_stds, name='uncertainties',
                            extra_data = dict(n_measured=len(self.inputs)),
                            )

        choice_indices = np.argsort(stds.flatten())
        num_most_uncertain_to_measure_first = 1
        most_uncertain_indices = choice_indices[-num_most_uncertain_to_measure_first:][::-1]
        p = stds[:-num_most_uncertain_to_measure_first]
        if p.sum()!=0:
            p /= p.sum()
            probabalistic_indices = np.random.choice(choice_indices[:-num_most_uncertain_to_measure_first], 
                                                size=batch_size-num_most_uncertain_to_measure_first,
                                                replace=False,
                                                p=p)
            selected_indices =  np.concatenate([most_uncertain_indices, probabalistic_indices])
        else:
            selected_indices = choice_indices[-batch_size:][::-1]
        
        next_positions = test_x[selected_indices]
        partial.apply_async(args=[], kwargs=dict(grid_points=self.meshgrid_points,
                            scatter_xs=next_positions.T[0],
                            scatter_ys=next_positions.T[1],
                            bounds=self.input_bounds,
                            filename='stds.png'))
        
        print("Next positions selected...")
        print("Plotting")
        grid_inputs = griddata(self.inputs.cpu(), self.targets.cpu(), (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        plt.imshow(grid_inputs, cmap='terrain', origin='lower')
        cb = plt.colorbar() 
        plt.savefig(f'clustering_outputs.png')
        cb.remove()
        grid_evaluations = griddata(test_x, evaluation.loc.max(0)[1].cpu(), (self.meshgrid_points[0], self.meshgrid_points[1]), method='nearest')
        tasks.image_report(experiment_id=self.experiment_id,
                           matrix=grid_evaluations, name='predictions',
                           extra_data = dict(n_measured=len(self.inputs))
                        )
        plt.scatter(*self.inputs.cpu().numpy().T, marker='.', s=1, c='r')
        plt.imshow(grid_evaluations, cmap='terrain', origin='lower', extent=np.ravel(self.input_bounds))
        cb = plt.colorbar() 
        plt.savefig(f'predictions.png')
        plt.clf()
        return next_positions

    def tell_many(self, x, y):
        raise NotImplementedError
