"""Estimador de profundidade métrica utilizando Ultralytics YOLO Depth (YOLO26-Depth).

Produz profundidade métrica contínua em metros a partir de uma imagem RGB monocular,
utilizando o pipeline oficial da Ultralytics.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from src.config import resolve_path
from src.inference.base_estimator import (
    BaseDepthEstimator,
    DepthOutputType,
    DepthResult,
    EstimatorError,
    synchronize,
    validate_rgb_image,
)


class YOLODepthEstimator(BaseDepthEstimator):
    """Estimador de profundidade métrica baseado em Ultralytics YOLO Depth."""

    def __init__(self, model_key: str, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(model_key, config, **kwargs)
        if self.output_type is not DepthOutputType.METRIC_DEPTH:
            raise EstimatorError(f"{model_key}: YOLO Depth produz apenas profundidade métrica.")
        self.weights_path: str = self.config.get("weights_path", "models/cache/yolo/yolo26n-depth.pt")
        self.model_name_cfg: str = self.config.get("model_name", "yolo26n-depth.pt")
        self.imgsz: int = int(self.config.get("imgsz", 768))
        self.model: Any = None

    def _load(self) -> None:
        # Previne erros de permissão em diretórios de configuração do usuário dentro do sandbox
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise EstimatorError("Pacote 'ultralytics' não instalado. Instale via 'pip install ultralytics'.") from exc

        resolved = resolve_path(self.weights_path)
        weights_target = str(resolved) if resolved.is_file() else self.model_name_cfg

        try:
            self.model = YOLO(weights_target)
            target_device = str(self.device)
            self.model.to(target_device)
        except Exception as exc:  # noqa: BLE001
            raise EstimatorError(f"Falha ao carregar YOLO Depth ({weights_target}): {exc}") from exc

    def _preprocess(self, image: np.ndarray) -> Any:
        # Método abstrato da base; a pipeline completa é executada em predict()
        return image

    def _forward(self, inputs: Any) -> torch.Tensor:
        # Método abstrato da base
        return torch.empty(0)

    def _input_shape(self, inputs: Any) -> tuple[int, ...]:
        return (1, 3, self.imgsz, self.imgsz)

    @torch.inference_mode()
    def predict(self, image: np.ndarray, warmup_runs: int = 0) -> DepthResult:
        """Executa a inferência de profundidade métrica usando Ultralytics."""
        validate_rgb_image(image)
        self.load()
        h, w = image.shape[:2]

        # Ultralytics espera imagens numpy em formato BGR
        bgr = image[:, :, ::-1].copy()
        target_device = str(self.device)

        # Warmup se solicitado
        for _ in range(max(0, warmup_runs)):
            self.model.predict(bgr, imgsz=self.imgsz, device=target_device, verbose=False)

        synchronize(self.device)
        t_start = time.perf_counter()

        # Execução completa (preprocessamento, inferência da rede e postprocessamento de unpadding)
        results = self.model.predict(bgr, imgsz=self.imgsz, device=target_device, verbose=False)
        synchronize(self.device)
        t_total = time.perf_counter() - t_start

        if not results or results[0].depth is None:
            raise EstimatorError(f"{self.model_key}: modelo não retornou mapa de profundidade válido.")

        r = results[0]
        # Mapa métrico final em metros interpolado e recortado para as dimensões exatas (H, W)
        depth_map = r.depth.data.cpu().numpy().astype(np.float32)

        # Tempos reportados internamente pela Ultralytics (em milissegundos)
        speed = getattr(r, "speed", {})
        prep_time = float(speed.get("preprocess", 0.0)) / 1000.0
        inf_time = float(speed.get("inference", t_total * 1000.0)) / 1000.0
        post_time = float(speed.get("postprocess", 0.0)) / 1000.0

        # Para compatibilidade com a interface e estatísticas, tenta obter a saída em baixa resolução
        try:
            im_tensor = self.model.predictor.preprocess([bgr])
            raw_tensor = self.model.model(im_tensor)
            raw_output = raw_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            input_shape = tuple(im_tensor.shape)
        except Exception:
            raw_output = depth_map
            input_shape = (1, 3, self.imgsz, self.imgsz)

        return DepthResult(
            model_key=self.model_key,
            model_name=self.model_name,
            depth=depth_map,
            raw_output=raw_output,
            output_type=self.output_type,
            units=self.units,
            image_size=(h, w),
            model_input_shape=input_shape,
            inference_time_s=inf_time,
            preprocess_time_s=prep_time,
            postprocess_time_s=post_time,
            device=target_device,
            config=self.describe() | {"imgsz": self.imgsz, "warmup_runs": warmup_runs},
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["framework"] = "ultralytics"
        info["task"] = "depth"
        info["imgsz"] = self.imgsz
        return info
