"""
Simple finetuning script for a VAE registered in MLflow.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import mlflow

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
BASE_MODEL_NAME = "smi_auto_vae"          # registered model to load
NEW_MODEL_NAME  = "smi_auto_vae_finetuned"
VAE_CODE_PATH   = "../models/vae_202507/vae.py"
DATA_DIR        = "/path/to/your/images"  # folder with PNG/NPY images
IMAGE_SIZE      = (512, 512)
LATENT_DIM      = 512
EPOCHS          = 10
LR              = 1e-4
BATCH_SIZE      = 16
KL_WEIGHT       = 0.001
FREEZE_ENCODER  = False
SAVE_DIR        = "/tmp"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, folder, image_size):
        self.paths = sorted(
            p for p in Path(folder).rglob("*")
            if p.suffix.lower() in (".png", ".tif", ".tiff", ".npy")
        )
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        arr = np.load(str(p)) if p.suffix == ".npy" else np.array(Image.open(str(p)))
        arr = arr.squeeze()
        # normalize to uint8
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        return self.transform(Image.fromarray(arr))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def vae_loss(x_recon, x, mu, logvar, kl_weight):
    recon = nn.functional.mse_loss(x_recon, x)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load the registered model from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    loaded  = mlflow.pyfunc.load_model(f"models:/{BASE_MODEL_NAME}/latest")
    wrapper = loaded._model_impl.python_model   # VAEModelWrapper instance
    model   = wrapper.model                     # the raw nn.Module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.train()

    # 2. Optionally freeze encoder
    if FREEZE_ENCODER:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — training decoder only.")

    # 3. Dataset & optimizer
    dataset   = ImageDataset(DATA_DIR, IMAGE_SIZE)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    print(f"Training on {len(dataset)} images for {EPOCHS} epochs ...")

    # 4. Training loop
    with mlflow.start_run(run_name=f"finetune_{NEW_MODEL_NAME}"):
        mlflow.log_params({
            "base_model": BASE_MODEL_NAME,
            "epochs": EPOCHS, "lr": LR,
            "batch_size": BATCH_SIZE, "kl_weight": KL_WEIGHT,
            "freeze_encoder": FREEZE_ENCODER,
        })

        for epoch in range(EPOCHS):
            total_loss = total_recon = total_kl = 0.0

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                x_recon, mu, logvar = model(batch)
                loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, KL_WEIGHT)

                loss.backward()
                optimizer.step()

                total_loss  += loss.item()
                total_recon += recon.item()
                total_kl    += kl.item()

            n = len(loader)
            print(f"Epoch {epoch+1}/{EPOCHS}  "
                  f"loss={total_loss/n:.5f}  "
                  f"recon={total_recon/n:.5f}  "
                  f"kl={total_kl/n:.5f}")
            mlflow.log_metrics({
                "train_loss": total_loss / n,
                "recon_loss": total_recon / n,
                "kl_loss":    total_kl / n,
            }, step=epoch)

    # 5. Save weights as NPZ (same format your wrappers expect)
    weights_path = os.path.join(SAVE_DIR, f"{NEW_MODEL_NAME}_weights.npz")
    np.savez(weights_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
    print(f"Weights saved to {weights_path}")

    # 6. Re-register the finetuned model using your existing wrapper
    from vae_wrapper import save_vae_model_with_wrapper
    save_vae_model_with_wrapper(
        model_config={
            "name":         NEW_MODEL_NAME,
            "state_dict":   weights_path,
            "python_file":  VAE_CODE_PATH,
            "python_class": "ConvVAE",
            "type":         "torch",
            "latent_dim":   LATENT_DIM,
            "image_size":   IMAGE_SIZE,
        },
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name="finetune_experiment",
        model_name=NEW_MODEL_NAME,
    )
    print(f"✅ Finetuned model registered as '{NEW_MODEL_NAME}'")
