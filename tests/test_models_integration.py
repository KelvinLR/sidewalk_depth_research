"""Testes com os modelos REAIS (marcados como ``slow``).

Na primeira execução baixam os pesos (~330 MB, exige internet); depois usam o
cache em ``models/cache/``. Para pular: ``pytest -m "not slow"``.

Usam uma imagem sintética: verificam carregamento, formatos e semântica da
saída — não a qualidade da profundidade.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import DepthOutputType, create_estimator

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    h, w = 240, 320
    grad = np.linspace(0, 255, h, dtype=np.float32)[:, None, None].repeat(w, 1).repeat(3, 2)
    return np.clip(grad + rng.normal(0, 10, (h, w, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "key,output_type,units",
    [
        ("depth_anything_v2_small", DepthOutputType.RELATIVE_INVERSE_DEPTH, None),
        ("depth_anything_v2_metric_outdoor_small", DepthOutputType.METRIC_DEPTH, "meters"),
        ("midas_v21_small", DepthOutputType.RELATIVE_INVERSE_DEPTH, None),
    ],
)
def test_real_model_inference_cpu(key, output_type, units, synthetic_image) -> None:
    result = create_estimator(key, device="cpu").predict(synthetic_image)
    assert result.depth.shape == synthetic_image.shape[:2]
    assert result.output_type is output_type and result.units == units
    assert np.isfinite(result.raw_output).all()
    assert result.inference_time_s > 0
    if output_type is DepthOutputType.METRIC_DEPTH:
        # Cabeça métrica = sigmoid * max_depth -> valores em (0, 80) m.
        assert 0 < result.raw_output.min() and result.raw_output.max() < 80
    else:
        assert result.raw_output.min() >= 0  # cabeças relativas são não negativas (ReLU)
