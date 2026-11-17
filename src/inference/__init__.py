"""Carregamento e execução de modelos de estimativa de profundidade."""

from src.inference.base_estimator import (
    BaseDepthEstimator,
    DepthOutputType,
    DepthResult,
    EstimatorError,
    resolve_device,
)
from src.inference.model_factory import available_models, create_estimator
from src.inference.results_io import append_inference_log, load_depth_result, save_depth_result

__all__ = [
    "BaseDepthEstimator",
    "DepthOutputType",
    "DepthResult",
    "EstimatorError",
    "append_inference_log",
    "available_models",
    "create_estimator",
    "load_depth_result",
    "resolve_device",
    "save_depth_result",
]
