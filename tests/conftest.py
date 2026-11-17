"""Fixtures compartilhadas."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # sem janelas durante os testes

import numpy as np
import pytest
import torch

from src.inference.base_estimator import BaseDepthEstimator, DepthOutputType


class DummyEstimator(BaseDepthEstimator):
    """Estimador sintético: saída = média dos canais reduzida pela metade.

    Permite testar o fluxo da classe-base sem baixar pesos.
    """

    def _load(self) -> None:
        self.load_calls = getattr(self, "load_calls", 0) + 1

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(image).permute(2, 0, 1).float()[None]
        return torch.nn.functional.avg_pool2d(t, 2)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=1)  # (1, h/2, w/2)

    def _input_shape(self, inputs: Any) -> tuple[int, ...]:
        return tuple(inputs.shape)


@pytest.fixture
def dummy_config() -> dict[str, Any]:
    return {"display_name": "Dummy", "output_type": DepthOutputType.RELATIVE_INVERSE_DEPTH.value, "units": None}


@pytest.fixture
def dummy_estimator(dummy_config) -> DummyEstimator:
    return DummyEstimator("dummy", dummy_config, device="cpu")


@pytest.fixture
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
