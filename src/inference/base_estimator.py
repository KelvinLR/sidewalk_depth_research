"""Interface comum dos estimadores de profundidade.

A classe-base padroniza apenas o *fluxo* de execução e o *formato do resultado*.
Cada modelo mantém seu próprio pré-processamento, carregamento e semântica de
saída — as diferenças entre modelos são expostas em :class:`DepthResult`, não
escondidas.

Fluxo de :meth:`BaseDepthEstimator.predict`::

    imagem RGB uint8 (H, W, 3)
        -> _preprocess()        específico do modelo
        -> _forward()           inferência (tempo medido isoladamente)
        -> saída bruta na resolução do modelo (h, w), float32, SEM normalização
        -> interpolação para (H, W)  (mesmo método para todos os modelos)
        -> DepthResult
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class DepthOutputType(str, Enum):
    """Natureza física da saída de um modelo."""

    #: Disparidade / inverso de profundidade, afim-invariante (escala e
    #: deslocamento desconhecidos). MAIOR = MAIS PERTO. Sem unidade física.
    RELATIVE_INVERSE_DEPTH = "relative_inverse_depth"
    #: Profundidade métrica (distância ao longo do eixo óptico). MAIOR = MAIS LONGE.
    METRIC_DEPTH = "metric_depth"

    @property
    def larger_is_closer(self) -> bool:
        return self is DepthOutputType.RELATIVE_INVERSE_DEPTH

    @property
    def description(self) -> str:
        if self is DepthOutputType.RELATIVE_INVERSE_DEPTH:
            return "Disparidade relativa (inverso de profundidade, sem unidade; maior = mais perto)"
        return "Profundidade métrica (maior = mais longe)"


class EstimatorError(RuntimeError):
    """Falha ao carregar ou executar um estimador."""


@dataclass
class DepthResult:
    """Resultado de uma inferência.

    Attributes:
        model_key: chave do modelo em ``settings.yaml``.
        model_name: nome legível do modelo.
        depth: saída na RESOLUÇÃO ORIGINAL da imagem ``(H, W)``, ``float32``,
            obtida por interpolação da saída bruta. NÃO normalizada.
        raw_output: saída bruta do modelo na resolução de inferência ``(h, w)``,
            ``float32``, exatamente como produzida pela rede.
        output_type: natureza da saída (relativa ou métrica).
        units: unidade física (``"meters"``) ou ``None`` para saídas relativas.
        image_size: ``(H, W)`` da imagem de entrada.
        model_input_shape: shape do tensor efetivamente enviado à rede
            (ex.: ``(1, 3, 518, 686)``).
        inference_time_s: tempo apenas do forward pass (com sincronização do device).
        preprocess_time_s / postprocess_time_s: tempos das demais etapas.
        device: device utilizado (``cpu``, ``cuda``, ``mps``).
        config: configurações utilizadas (repositório, revisão, pré-processamento...).
        created_at: timestamp ISO-8601 (UTC).
    """

    model_key: str
    model_name: str
    depth: np.ndarray
    raw_output: np.ndarray
    output_type: DepthOutputType
    units: str | None
    image_size: tuple[int, int]
    model_input_shape: tuple[int, ...]
    inference_time_s: float
    preprocess_time_s: float
    postprocess_time_s: float
    device: str
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape da saída bruta do modelo (resolução de inferência)."""
        return tuple(self.raw_output.shape)  # type: ignore[return-value]

    @property
    def is_metric(self) -> bool:
        return self.output_type is DepthOutputType.METRIC_DEPTH

    @property
    def total_time_s(self) -> float:
        return self.preprocess_time_s + self.inference_time_s + self.postprocess_time_s

    def stats(self) -> dict[str, float]:
        """Estatísticas da saída bruta (resolução do modelo), ignorando NaN/inf."""
        values = self.raw_output[np.isfinite(self.raw_output)]
        if values.size == 0:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan")}
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }

    def to_metadata(self) -> dict[str, Any]:
        """Metadados serializáveis (sem os arrays)."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "output_type": self.output_type.value,
            "output_description": self.output_type.description,
            "larger_is_closer": self.output_type.larger_is_closer,
            "units": self.units,
            "image_size_hw": list(self.image_size),
            "model_input_shape": list(self.model_input_shape),
            "raw_output_shape_hw": list(self.output_shape),
            "inference_time_s": self.inference_time_s,
            "preprocess_time_s": self.preprocess_time_s,
            "postprocess_time_s": self.postprocess_time_s,
            "device": self.device,
            "raw_output_stats": self.stats(),
            "config": self.config,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def resolve_device(preference: str | torch.device | None = "auto") -> torch.device:
    """Resolve o device de execução.

    ``"auto"`` (ou ``None``) escolhe ``cuda`` > ``mps`` > ``cpu``. Um device
    explícito indisponível gera :class:`EstimatorError` em vez de cair
    silenciosamente para CPU.
    """
    if isinstance(preference, torch.device):
        preference = str(preference)
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise EstimatorError("CUDA solicitado, mas não está disponível neste ambiente.")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise EstimatorError("MPS solicitado, mas não está disponível neste ambiente.")
    try:
        return torch.device(pref)
    except RuntimeError as exc:
        raise EstimatorError(f"Device inválido: {preference!r}") from exc


def synchronize(device: torch.device) -> None:
    """Aguarda operações assíncronas do device (necessário para medir tempo)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resize_depth(raw: np.ndarray, size_hw: tuple[int, int], mode: str = "bicubic") -> np.ndarray:
    """Interpola um mapa 2D para ``size_hw`` sem alterar sua escala.

    Usa ``torch.nn.functional.interpolate`` (``align_corners=False``), o mesmo
    procedimento do pós-processamento oficial do Depth Anything (Transformers) e
    dos exemplos do MiDaS. A interpolação bicúbica pode gerar pequenos
    *overshoots* nas bordas (valores ligeiramente fora do intervalo original);
    use ``"bilinear"`` se isso for relevante.
    """
    if raw.ndim != 2:
        raise ValueError(f"Esperado mapa 2D, recebido shape {raw.shape}")
    if tuple(raw.shape) == tuple(size_hw):
        return raw.astype(np.float32, copy=True)
    tensor = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))[None, None]
    kwargs: dict[str, Any] = {} if mode in {"nearest", "area"} else {"align_corners": False}
    out = F.interpolate(tensor, size=tuple(size_hw), mode=mode, **kwargs)
    return out[0, 0].numpy().astype(np.float32, copy=False)


def validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """Valida uma imagem RGB ``uint8`` ``(H, W, 3)``."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Esperado np.ndarray, recebido {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Esperada imagem RGB (H, W, 3), recebido shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Esperado dtype uint8 (0–255), recebido {image.dtype}")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"Imagem muito pequena: {image.shape}")
    return image


# ---------------------------------------------------------------------------
# Classe-base
# ---------------------------------------------------------------------------


class BaseDepthEstimator(ABC):
    """Classe-base dos estimadores.

    Subclasses implementam :meth:`_load`, :meth:`_preprocess` e :meth:`_forward`.
    O modelo é carregado de forma preguiçosa na primeira chamada de
    :meth:`predict` (ou explicitamente via :meth:`load`).
    """

    def __init__(
        self,
        model_key: str,
        config: dict[str, Any],
        device: str | torch.device | None = "auto",
        interpolation: str = "bicubic",
    ) -> None:
        self.model_key = model_key
        self.config = dict(config)
        self.model_name: str = self.config.get("display_name", model_key)
        self.output_type = DepthOutputType(self.config["output_type"])
        self.units: str | None = self.config.get("units")
        if self.output_type is DepthOutputType.RELATIVE_INVERSE_DEPTH and self.units is not None:
            raise EstimatorError(f"{model_key}: saída relativa não pode declarar unidade física ({self.units}).")
        self.device = resolve_device(device)
        self.interpolation = interpolation
        self._loaded = False

    # -- API pública ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> BaseDepthEstimator:
        """Carrega pesos (usa o cache local; baixa apenas na primeira vez)."""
        if not self._loaded:
            try:
                self._load()
            except EstimatorError:
                raise
            except Exception as exc:  # noqa: BLE001 - reempacota com contexto
                raise EstimatorError(f"Falha ao carregar '{self.model_key}': {type(exc).__name__}: {exc}") from exc
            self._loaded = True
        return self

    @torch.inference_mode()
    def predict(self, image: np.ndarray, warmup_runs: int = 0) -> DepthResult:
        """Executa a inferência em uma imagem RGB ``uint8`` ``(H, W, 3)``.

        Args:
            image: imagem RGB (use ``src.preprocessing.load_rgb_image``).
            warmup_runs: execuções descartadas antes da medição (a primeira
                inferência costuma ser mais lenta, especialmente em GPU/MPS).
        """
        validate_rgb_image(image)
        self.load()
        h, w = image.shape[:2]

        t0 = time.perf_counter()
        inputs = self._preprocess(image)
        synchronize(self.device)
        preprocess_time = time.perf_counter() - t0

        try:
            for _ in range(max(0, warmup_runs)):
                self._forward(inputs)
            synchronize(self.device)
            t1 = time.perf_counter()
            raw = self._forward(inputs)
            synchronize(self.device)
            inference_time = time.perf_counter() - t1
        except Exception as exc:  # noqa: BLE001
            raise EstimatorError(f"Falha na inferência de '{self.model_key}': {type(exc).__name__}: {exc}") from exc

        t2 = time.perf_counter()
        raw_np = self._to_2d_numpy(raw)
        depth = resize_depth(raw_np, (h, w), mode=self.interpolation)
        postprocess_time = time.perf_counter() - t2

        return DepthResult(
            model_key=self.model_key,
            model_name=self.model_name,
            depth=depth,
            raw_output=raw_np,
            output_type=self.output_type,
            units=self.units,
            image_size=(h, w),
            model_input_shape=tuple(self._input_shape(inputs)),
            inference_time_s=inference_time,
            preprocess_time_s=preprocess_time,
            postprocess_time_s=postprocess_time,
            device=str(self.device),
            config=self.describe() | {"warmup_runs": warmup_runs, "interpolation": self.interpolation},
        )

    def describe(self) -> dict[str, Any]:
        """Configurações relevantes para reprodutibilidade (sobrescrever para detalhar)."""
        return {k: v for k, v in self.config.items() if k != "enabled"}

    def __repr__(self) -> str:
        state = "carregado" if self._loaded else "não carregado"
        return f"{type(self).__name__}({self.model_key!r}, device={self.device}, {state})"

    # -- Pontos de extensão --------------------------------------------------

    @abstractmethod
    def _load(self) -> None:
        """Carrega o modelo e o pré-processador."""

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> Any:
        """Converte a imagem RGB uint8 nas entradas específicas do modelo."""

    @abstractmethod
    def _forward(self, inputs: Any) -> torch.Tensor:
        """Executa a rede e retorna a saída bruta (``(1, h, w)`` ou ``(h, w)``)."""

    @abstractmethod
    def _input_shape(self, inputs: Any) -> tuple[int, ...]:
        """Shape do tensor enviado à rede."""

    # -- Utilidades ----------------------------------------------------------

    @staticmethod
    def _to_2d_numpy(raw: torch.Tensor) -> np.ndarray:
        arr = raw.detach().to("cpu", torch.float32).squeeze()
        if arr.ndim != 2:
            raise EstimatorError(f"Saída inesperada do modelo: shape {tuple(raw.shape)}")
        return arr.numpy().copy()
