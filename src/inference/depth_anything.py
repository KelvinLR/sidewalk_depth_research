"""Depth Anything V2 via Hugging Face Transformers.

Uma única classe atende às duas variantes, porque elas compartilham arquitetura
(DINOv2 ViT-S/14 + cabeça DPT) e pré-processador (``DPTImageProcessor``). A
diferença está na **cabeça de saída**, declarada em ``config.json`` do modelo:

- ``Depth-Anything-V2-Small-hf``: ``depth_estimation_type`` ausente/``relative``
  -> disparidade relativa afim-invariante (maior = mais perto, sem unidade).
- ``Depth-Anything-V2-Metric-Outdoor-Small-hf``: ``depth_estimation_type = metric``,
  ``max_depth = 80`` -> profundidade em metros (``sigmoid * max_depth``),
  ajustada no Virtual KITTI 2 (cenas de rua sintéticas).

Ao carregar, a natureza declarada em ``settings.yaml`` é conferida contra o
``config.json`` do modelo; divergências geram erro.

Pré-processamento (``DPTImageProcessor``): lado menor redimensionado para 518 px
mantendo a proporção, dimensões múltiplas de 14 (patch do ViT), interpolação
bicúbica, normalização ImageNet, sem padding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from src.config import get_huggingface_cache_dir
from src.inference.base_estimator import BaseDepthEstimator, DepthOutputType, EstimatorError


class DepthAnythingV2Estimator(BaseDepthEstimator):
    """Estimador para Depth Anything V2 (Small relativo e Metric Outdoor Small)."""

    def __init__(self, model_key: str, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(model_key, config, **kwargs)
        if "hf_repo_id" not in self.config:
            raise EstimatorError(f"{model_key}: 'hf_repo_id' ausente na configuração.")
        self.repo_id: str = self.config["hf_repo_id"]
        self.revision: str = self.config.get("revision", "main")
        self.processor: Any = None
        self.model: Any = None

    def _load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        cache_dir = str(get_huggingface_cache_dir())
        self.processor = AutoImageProcessor.from_pretrained(
            self.repo_id, revision=self.revision, cache_dir=cache_dir
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            self.repo_id, revision=self.revision, cache_dir=cache_dir
        )
        self.model = model.to(self.device).eval()
        self._check_output_semantics()

    def _check_output_semantics(self) -> None:
        """Confere se a configuração local corresponde à cabeça do modelo."""
        declared = getattr(self.model.config, "depth_estimation_type", None) or "relative"
        expected = "metric" if self.output_type is DepthOutputType.METRIC_DEPTH else "relative"
        if declared != expected:
            raise EstimatorError(
                f"{self.model_key}: settings.yaml declara '{self.output_type.value}', mas o modelo "
                f"{self.repo_id} declara depth_estimation_type='{declared}'."
            )
        if self.output_type is DepthOutputType.METRIC_DEPTH:
            model_max = getattr(self.model.config, "max_depth", None)
            cfg_max = self.config.get("max_depth_m")
            if cfg_max is not None and model_max is not None and float(cfg_max) != float(model_max):
                raise EstimatorError(
                    f"{self.model_key}: max_depth_m={cfg_max} em settings.yaml difere de "
                    f"max_depth={model_max} no modelo."
                )

    def _preprocess(self, image: np.ndarray) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].to(self.device)}

    def _forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs).predicted_depth  # (1, h, w)

    def _input_shape(self, inputs: dict[str, torch.Tensor]) -> tuple[int, ...]:
        return tuple(inputs["pixel_values"].shape)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        if self.processor is not None:
            size = getattr(self.processor, "size", None)
            info["preprocessing"] = {
                "processor_class": type(self.processor).__name__,
                "size": dict(size) if hasattr(size, "keys") else str(size),
                "keep_aspect_ratio": getattr(self.processor, "keep_aspect_ratio", None),
                "ensure_multiple_of": getattr(self.processor, "ensure_multiple_of", None),
                "resample": str(getattr(self.processor, "resample", None)),
                "image_mean": list(getattr(self.processor, "image_mean", []) or []),
                "image_std": list(getattr(self.processor, "image_std", []) or []),
            }
        if self.model is not None:
            cfg = self.model.config
            info["model_config"] = {
                "depth_estimation_type": getattr(cfg, "depth_estimation_type", None) or "relative",
                "max_depth": getattr(cfg, "max_depth", None),
                "commit_hash": getattr(cfg, "_commit_hash", None),
            }
        return info
