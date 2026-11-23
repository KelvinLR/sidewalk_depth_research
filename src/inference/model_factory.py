"""Criação de estimadores a partir das chaves de ``config/settings.yaml``.

Exemplo::

    from src.inference import create_estimator
    estimator = create_estimator("depth_anything_v2_small")   # não carrega ainda
    result = estimator.predict(image_rgb)                     # carrega na 1ª chamada
"""

from __future__ import annotations

from typing import Any

import torch

from src.config import get_model_config, load_settings
from src.inference.base_estimator import BaseDepthEstimator, EstimatorError
from src.inference.depth_anything import DepthAnythingV2Estimator
from src.inference.midas import MiDaSEstimator
from src.inference.yolo_depth import YOLODepthEstimator

#: Mapeia o campo ``backend`` do settings.yaml para a classe que o implementa.
BACKENDS: dict[str, type[BaseDepthEstimator]] = {
    "transformers": DepthAnythingV2Estimator,
    "torch_hub": MiDaSEstimator,
    "ultralytics": YOLODepthEstimator,
}


def available_models(enabled_only: bool = True) -> list[str]:
    """Chaves de modelos configurados (por padrão, apenas os habilitados)."""
    models: dict[str, dict[str, Any]] = load_settings().get("models", {})
    return [k for k, cfg in models.items() if not enabled_only or cfg.get("enabled", True)]


def create_estimator(
    model_key: str,
    device: str | torch.device | None = None,
    interpolation: str | None = None,
) -> BaseDepthEstimator:
    """Instancia (sem carregar) o estimador de ``model_key``.

    Args:
        model_key: chave em ``settings.yaml > models``.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``. Padrão: ``runtime.device``.
        interpolation: método para voltar à resolução original. Padrão:
            ``preprocessing.output_interpolation`` (ou ``"bicubic"``).
    """
    config = get_model_config(model_key)
    backend = config.get("backend")
    if backend not in BACKENDS:
        raise EstimatorError(f"{model_key}: backend desconhecido {backend!r}. Opções: {sorted(BACKENDS)}")
    settings = load_settings()
    if device is None:
        device = settings.get("runtime", {}).get("device", "auto")
    if interpolation is None:
        interpolation = settings.get("preprocessing", {}).get("output_interpolation", "bicubic")
    return BACKENDS[backend](model_key, config, device=device, interpolation=interpolation)
