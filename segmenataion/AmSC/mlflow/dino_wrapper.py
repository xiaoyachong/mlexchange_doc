"""
DINOv3 MLflow wrapper — local checkpoint version.
Wraps lightly_train.load_model() + predict_batch_correct().

predict() input (dict or single-row DataFrame):
    {
        "images": np.ndarray  shape (N, C, H, W)  float32 [0, 1]
    }

predict() output:
    {
        "masks":   np.ndarray  shape (N, H, W)  uint8 class indices
        "success": bool
    }
"""

import logging
import os

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

logger = logging.getLogger("seg.dino_wrapper")


class DinoWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for fine-tuned DINOv3 (lightly_train checkpoint).
    predict_batch_correct() logic is inlined so inference_dino_v1.py needs
    no changes — it can keep using its own function or call this wrapper.
    """

    supports_batch = True

    # ------------------------------------------------------------------ #
    # load_context                                                         #
    # ------------------------------------------------------------------ #
    def _load_model(self, context):
        if os.getenv("MLFLOW_DISABLE_MODEL_LOADING", "false").lower() == "true":
            logger.info("Skipping model loading (registration mode)")
            return

        import lightly_train

        ckpt_path = context.artifacts["checkpoint"]
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        logger.info(f"Loading DINO model from {ckpt_path} on {device}")
        self.model = lightly_train.load_model(ckpt_path)
        self.model.eval().to(device)
        logger.info("DinoWrapper loaded successfully")

    def load_context(self, context):
        self._load_model(context)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    def _parse_input(self, model_input):
        if hasattr(model_input, "to_dict"):
            return model_input.to_dict("records")[0]
        return model_input

    def _run_inference(self, batch_images: np.ndarray) -> np.ndarray:
        """
        Mirrors predict_batch_correct() from inference_dino_v1.py exactly.

        Args:
            batch_images: float32 ndarray (N, C, H, W) in [0, 1]

        Returns:
            pred_masks: uint8 ndarray (N, H, W) with class indices
        """
        tensor = torch.from_numpy(batch_images).to(self.device)   # (N, C, H, W)
        pred_masks = []

        for i in range(tensor.shape[0]):
            x = tensor[i]                                          # (C, H, W)
            image_h, image_w = x.shape[-2:]

            if x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)

            # Normalize — same as predict_batch_correct
            x = T.functional.normalize(
                x,
                mean=self.model.image_normalize["mean"],
                std=self.model.image_normalize["std"],
            )

            # Resize to crop size (short side)
            crop_size = min(self.model.image_size)
            x = T.functional.resize(x, size=[crop_size])
            x = x.unsqueeze(0)                                     # (1, C, H', W')

            logits = self.model._forward_logits(x)                 # (1, K+1, H', W')
            logits = logits[:, :-1]                                # (1, K,   H', W')

            logits = F.interpolate(
                logits, size=(image_h, image_w), mode="bilinear"
            )                                                       # (1, K, H, W)

            masks = logits.argmax(dim=1)                           # (1, H, W)
            masks = self.model.internal_class_to_class[masks]      # (1, H, W)
            pred_masks.append(masks[0].cpu().numpy().astype(np.uint8))

        return np.stack(pred_masks, axis=0)                        # (N, H, W)

    # ------------------------------------------------------------------ #
    # predict — supports single (N=1) and batch (N>1)                     #
    # ------------------------------------------------------------------ #
    def predict(self, context, model_input):
        """
        Input dict keys:
            images : np.ndarray  (N, C, H, W)  float32 [0, 1]
                     N=1 for single inference, N>1 for batch

        Output dict keys:
            masks   : np.ndarray  (N, H, W)  uint8
            success : bool
            error   : str  (only on failure)
        """
        try:
            data   = self._parse_input(model_input)
            images = data["images"]                                # (N, C, H, W)

            if not isinstance(images, np.ndarray):
                images = np.array(images, dtype=np.float32)

            # Ensure 4-D: treat a single (C, H, W) as batch of 1
            if images.ndim == 3:
                images = images[np.newaxis]                        # (1, C, H, W)

            if images.dtype != np.float32:
                images = images.astype(np.float32)

            with torch.inference_mode():
                masks = self._run_inference(images)                # (N, H, W)

            return {"masks": masks, "success": True}

        except Exception as e:
            logger.error(f"DinoWrapper predict error: {e}", exc_info=True)
            return {"masks": np.array([]), "success": False, "error": str(e)}
