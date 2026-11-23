"""Carregamento e execução de modelos de estimativa de profundidade."""

from src.inference.base_estimator import (
    BaseDepthEstimator,
    DepthOutputType,
    DepthResult,
    EstimatorError,
    resolve_device,
)
from src.inference.batch_runner import BatchInferenceSummary, run_batch_inference
from src.inference.model_factory import available_models, create_estimator
from src.inference.results_io import append_inference_log, load_depth_result, save_depth_result
from src.inference.yolo_depth import YOLODepthEstimator

__all__ = [
    "BaseDepthEstimator",
    "BatchInferenceSummary",
    "DepthOutputType",
    "DepthResult",
    "EstimatorError",
    "YOLODepthEstimator",
    "append_inference_log",
    "available_models",
    "create_estimator",
    "load_depth_result",
    "resolve_device",
    "run_batch_inference",
    "save_depth_result",
]
