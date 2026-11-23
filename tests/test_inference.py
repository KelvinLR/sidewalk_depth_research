"""Testes da classe-base, da factory e da persistência (sem baixar pesos)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.inference import (
    DepthOutputType,
    EstimatorError,
    append_inference_log,
    available_models,
    create_estimator,
    load_depth_result,
    resolve_device,
    save_depth_result,
)
from src.inference.base_estimator import resize_depth, validate_rgb_image
from src.inference.depth_anything import DepthAnythingV2Estimator
from src.inference.midas import MiDaSEstimator
from src.inference.yolo_depth import YOLODepthEstimator
from tests.conftest import DummyEstimator

# --- classe-base -------------------------------------------------------------


def test_predict_restores_original_resolution_and_keeps_raw(dummy_estimator, rgb_image) -> None:
    result = dummy_estimator.predict(rgb_image)
    assert result.depth.shape == (60, 80)
    assert result.raw_output.shape == (30, 40)  # resolução do "modelo"
    assert result.output_shape == (30, 40)
    assert result.depth.dtype == np.float32 and result.raw_output.dtype == np.float32
    assert result.model_input_shape == (1, 3, 30, 40)
    assert result.image_size == (60, 80)


def test_raw_output_is_not_normalized(dummy_estimator, rgb_image) -> None:
    result = dummy_estimator.predict(rgb_image)
    # A saída do dummy está na escala 0–255; normalização a teria levado a [0, 1].
    assert result.raw_output.max() > 1.5
    expected = rgb_image.reshape(30, 2, 40, 2, 3).mean(axis=(1, 3, 4))
    np.testing.assert_allclose(result.raw_output, expected, rtol=1e-5)


def test_timings_and_metadata(dummy_estimator, rgb_image) -> None:
    result = dummy_estimator.predict(rgb_image, warmup_runs=2)
    assert result.inference_time_s >= 0 and result.preprocess_time_s >= 0
    meta = result.to_metadata()
    assert meta["output_type"] == "relative_inverse_depth"
    assert meta["larger_is_closer"] is True
    assert meta["units"] is None
    assert meta["config"]["warmup_runs"] == 2
    assert meta["device"] == "cpu"
    assert set(meta["raw_output_stats"]) == {"min", "max", "mean", "median"}


def test_model_is_loaded_once(dummy_estimator, rgb_image) -> None:
    dummy_estimator.predict(rgb_image)
    dummy_estimator.predict(rgb_image)
    assert dummy_estimator.load_calls == 1


def test_relative_output_cannot_declare_units(dummy_config) -> None:
    with pytest.raises(EstimatorError):
        DummyEstimator("bad", dummy_config | {"units": "meters"}, device="cpu")


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_validate_rgb_image_rejects_invalid(bad) -> None:
    with pytest.raises(ValueError):
        validate_rgb_image(bad)


def test_output_type_semantics() -> None:
    assert DepthOutputType.RELATIVE_INVERSE_DEPTH.larger_is_closer
    assert not DepthOutputType.METRIC_DEPTH.larger_is_closer


def test_resize_depth_preserves_scale() -> None:
    raw = np.full((10, 20), 7.5, dtype=np.float32)
    out = resize_depth(raw, (40, 80))
    assert out.shape == (40, 80)
    np.testing.assert_allclose(out, 7.5, rtol=1e-6)


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


# --- factory -----------------------------------------------------------------


def test_available_models() -> None:
    assert set(available_models()) == {
        "depth_anything_v2_small",
        "depth_anything_v2_metric_outdoor_small",
        "midas_v21_small",
        "yolo26n_depth",
    }


@pytest.mark.parametrize(
    "key,cls,output_type",
    [
        ("depth_anything_v2_small", DepthAnythingV2Estimator, DepthOutputType.RELATIVE_INVERSE_DEPTH),
        ("depth_anything_v2_metric_outdoor_small", DepthAnythingV2Estimator, DepthOutputType.METRIC_DEPTH),
        ("midas_v21_small", MiDaSEstimator, DepthOutputType.RELATIVE_INVERSE_DEPTH),
        ("yolo26n_depth", YOLODepthEstimator, DepthOutputType.METRIC_DEPTH),
    ],
)
def test_factory_creates_lazy_estimators(key, cls, output_type) -> None:
    est = create_estimator(key, device="cpu")
    assert isinstance(est, cls)
    assert est.output_type is output_type
    assert not est.is_loaded  # nenhum peso carregado


# --- persistência ------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path, dummy_estimator, rgb_image) -> None:
    result = dummy_estimator.predict(rgb_image)
    image_path = tmp_path / "sidewalk.png"
    image_path.write_bytes(b"fake image bytes")
    npz = save_depth_result(result, image_path, base_dir=tmp_path / "out")
    assert npz == tmp_path / "out" / "sidewalk" / "dummy.npz"

    arrays, meta = load_depth_result(npz)
    np.testing.assert_array_equal(arrays["depth"], result.depth)
    np.testing.assert_array_equal(arrays["raw_output"], result.raw_output)
    assert meta["output_type"] is DepthOutputType.RELATIVE_INVERSE_DEPTH
    assert meta["image"]["sha256"] and len(meta["image"]["sha256"]) == 64


def test_inference_log_appends(tmp_path, dummy_estimator, rgb_image) -> None:
    result = dummy_estimator.predict(rgb_image)
    log = tmp_path / "log.csv"
    append_inference_log(result, tmp_path / "missing.jpg", log_path=log)
    append_inference_log(result, tmp_path / "missing.jpg", log_path=log)
    with log.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["model_key"] == "dummy"
    assert rows[0]["output_type"] == "relative_inverse_depth"
