"""Testes para o processador em lote (src.inference.batch_runner)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.inference.base_estimator import DepthOutputType, DepthResult
from src.inference.batch_runner import BatchInferenceSummary, run_batch_inference


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Cria 2 imagens sintéticas para teste em lote."""
    imgs = []
    for name in ("test_sidewalk_1.jpg", "test_sidewalk_2.jpg"):
        p = tmp_path / name
        Image.new("RGB", (60, 40), color=(120, 120, 120)).save(p)
        imgs.append(p)
    return imgs


def test_batch_runner_validation() -> None:
    with pytest.raises(ValueError, match="vazia"):
        run_batch_inference(images=[])

    with pytest.raises(FileNotFoundError):
        run_batch_inference(images=["/caminho/inexistente/foto.jpg"])

    with pytest.raises(ValueError, match="Nenhum modelo"):
        run_batch_inference(images=["datasets/images/uneven_surfaces/buraco.jpg"], models=[])


def test_batch_runner_execution(sample_images: list[Path], tmp_path: Path, monkeypatch) -> None:
    # Redireciona paths de saída para tmp_path
    monkeypatch.setenv("SIDEWALK_OUTPUTS_DIR", str(tmp_path / "outputs"))

    # Mock do factory para evitar baixar/executar modelos pesados no teste unitário
    mock_estimator = MagicMock()
    dummy_depth = np.full((40, 60), 3.0, dtype=np.float32)
    dummy_result = DepthResult(
        model_key="mock_model",
        model_name="Mock Depth Small",
        depth=dummy_depth,
        raw_output=dummy_depth,
        output_type=DepthOutputType.RELATIVE_INVERSE_DEPTH,
        units=None,
        image_size=(40, 60),
        model_input_shape=(1, 3, 40, 60),
        inference_time_s=0.01,
        preprocess_time_s=0.005,
        postprocess_time_s=0.005,
        device="cpu",
    )
    mock_estimator.predict.return_value = dummy_result
    mock_estimator.load.return_value = mock_estimator

    with patch("src.inference.batch_runner.create_estimator", return_value=mock_estimator):
        summary = run_batch_inference(
            images=sample_images,
            models=["mock_model"],
            device="cpu",
            save_outputs=True,
            save_comparisons=True,
            save_defect_inspection=True,
            verbose=False,
        )

    assert isinstance(summary, BatchInferenceSummary)
    assert len(summary.results_by_image) == 2
    assert "test_sidewalk_1" in summary.results_by_image
    assert "test_sidewalk_2" in summary.results_by_image

    # Verifica o DataFrame de resumo
    assert len(summary.summary_df) == 2
    assert list(summary.summary_df["Imagem"]) == ["test_sidewalk_1.jpg", "test_sidewalk_2.jpg"]
    assert "Tempo Inferência (ms)" in summary.summary_df.columns

    # Verifica se o modelo foi carregado apenas uma vez
    mock_estimator.load.assert_called_once()
    # E predict chamado duas vezes (1 por imagem)
    assert mock_estimator.predict.call_count == 2
