"""
SAM3 MLflow wrapper — local checkpoint version.
Wraps only model load + single forward pass.
Patching / stitching / distributed logic stays in inference_v6.py.

predict() input (numpy structured as dict-in-DataFrame or plain dict):
    {
        "datapoints": <List[Datapoint]>  # already transformed by caller's ComposeAPI
    }

predict() output:
    {
        "processed_results": <dict>   # raw output of postprocessor.process_results()
        "success": bool
    }
"""

import logging
import os

import mlflow
import numpy as np
import torch

logger = logging.getLogger("seg.sam3_wrapper")


class SAM3Wrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for fine-tuned SAM3 (local checkpoint).
    Wraps model load and a single batched forward pass.
    All patching, transform, stitching stays in inference_v6.py.
    """

    supports_batch = True

    # ------------------------------------------------------------------ #
    # load_context                                                         #
    # ------------------------------------------------------------------ #
    def _load_model(self, context):
        if os.getenv("MLFLOW_DISABLE_MODEL_LOADING", "false").lower() == "true":
            logger.info("Skipping model loading (registration mode)")
            return

        from sam3 import build_sam3_image_model
        from sam3.eval.postprocessors import PostProcessImage
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI,
        )

        bpe_path            = context.artifacts["bpe_path"]
        original_ckpt       = context.artifacts["original_checkpoint"]
        finetuned_ckpt      = context.artifacts.get("finetuned_checkpoint")  # optional

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        logger.info(f"Loading SAM3 on {device}")

        # --- build base model ---
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
        )

        # --- load original weights ---
        self._load_state_dict_flexible(original_ckpt, strict=False)
        logger.info(f"Loaded original checkpoint: {original_ckpt}")

        # --- overlay finetuned weights if provided ---
        if finetuned_ckpt and os.path.exists(finetuned_ckpt):
            self._load_state_dict_flexible(finetuned_ckpt, strict=False)
            logger.info(f"Loaded finetuned checkpoint: {finetuned_ckpt}")

        self.model.eval().to(device)

        # --- default postprocessor (caller can override via input) ---
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=0.5,
            to_cpu=False,
        )

        # Store transform factory so caller can recreate if needed
        self._transform_cls = (ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
        logger.info("SAM3Wrapper loaded successfully")

    def load_context(self, context):
        self._load_model(context)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    def _load_state_dict_flexible(self, ckpt_path: str, strict: bool = False):
        """Load checkpoint, handling model/state_dict nesting and DDP prefix."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
        # strip DDP prefix
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    def _parse_input(self, model_input):
        """
        Accept:
          - plain dict  {"datapoints": [...], "confidence": 0.5}
          - pandas DataFrame with one row (MLflow REST path)
        Returns a plain dict.
        """
        if hasattr(model_input, "to_dict"):
            # DataFrame → take first record
            return model_input.to_dict("records")[0]
        return model_input

    # ------------------------------------------------------------------ #
    # predict                                                              #
    # ------------------------------------------------------------------ #
    def predict(self, context, model_input):
        """
        Single and batch inference share the same code path.

        Input dict keys:
            datapoints  : List[Datapoint]  — already transformed by ComposeAPI
            confidence  : float (optional, default 0.5)

        Output dict keys:
            processed_results : dict  keyed by query_id  (postprocessor output)
            success           : bool
            error             : str   (only on failure)
        """
        try:
            data = self._parse_input(model_input)
            datapoints  = data["datapoints"]          # List[Datapoint]
            confidence  = data.get("confidence", 0.5)

            # Update postprocessor threshold if different from default
            if confidence != self.postprocessor.detection_threshold:
                self.postprocessor.detection_threshold = confidence

            from sam3.train.data.collator import collate_fn_api as collate
            from sam3.model.utils.misc import copy_data_to_device

            batch = collate(datapoints, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            with torch.inference_mode():
                output = self.model(batch)

            processed_results = self.postprocessor.process_results(
                output, batch.find_metadatas
            )

            # Move tensors to CPU / numpy for serialisation
            serialisable = {}
            for qid, res in processed_results.items():
                entry = {}
                for k, v in res.items():
                    if torch.is_tensor(v):
                        entry[k] = v.cpu().numpy()
                    else:
                        entry[k] = v
                serialisable[qid] = entry

            return {"processed_results": serialisable, "success": True}

        except Exception as e:
            logger.error(f"SAM3Wrapper predict error: {e}", exc_info=True)
            return {"processed_results": {}, "success": False, "error": str(e)}
