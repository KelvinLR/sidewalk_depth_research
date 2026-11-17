"""Testes de leitura de imagem e de visualização."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src.inference import DepthOutputType
from src.preprocessing import ImageLoadError, list_dataset_images, load_rgb_image
from src.visualization import (
    display_range,
    normalize_for_display,
    plot_comparison,
    save_depth_visualizations,
    to_color_image,
    to_grayscale_image,
)

# --- preprocessing -----------------------------------------------------------


def test_load_rgb_jpeg(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    Image.new("RGB", (40, 20), (10, 200, 30)).save(path)
    img = load_rgb_image(path)
    assert img.rgb.shape == (20, 40, 3) and img.rgb.dtype == np.uint8
    assert img.category is None
    assert not img.orientation_applied


def test_exif_orientation_is_applied(tmp_path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # 90° horário -> largura e altura trocam
    Image.new("RGB", (40, 20), (100, 100, 100)).save(path, exif=exif)

    img = load_rgb_image(path, apply_exif_orientation=True)
    assert img.exif_orientation == 6 and img.orientation_applied
    assert img.size_hw == (40, 20)

    raw = load_rgb_image(path, apply_exif_orientation=False)
    assert raw.size_hw == (20, 40) and not raw.orientation_applied


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_non_rgb_modes_are_converted(tmp_path, mode) -> None:
    path = tmp_path / f"img_{mode}.png"
    Image.new(mode, (16, 8)).save(path)
    img = load_rgb_image(path)
    assert img.rgb.shape == (8, 16, 3)
    assert img.original_mode == mode


def test_missing_and_invalid_files(tmp_path) -> None:
    with pytest.raises(ImageLoadError):
        load_rgb_image(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(ImageLoadError):
        load_rgb_image(bad)


def test_list_dataset_images_validates_category() -> None:
    assert isinstance(list_dataset_images(), list)
    with pytest.raises(ValueError):
        list_dataset_images("unknown_category")


# --- visualization -----------------------------------------------------------


@pytest.fixture
def depth_map() -> np.ndarray:
    return np.linspace(1.0, 10.0, 200, dtype=np.float32).reshape(10, 20)


def test_normalize_does_not_mutate_input(depth_map) -> None:
    original = depth_map.copy()
    normalize_for_display(depth_map, DepthOutputType.METRIC_DEPTH)
    np.testing.assert_array_equal(depth_map, original)


def test_near_is_bright_convention(depth_map) -> None:
    # Relativa: maior valor = mais perto -> mais claro.
    rel = normalize_for_display(depth_map, DepthOutputType.RELATIVE_INVERSE_DEPTH)
    assert rel[-1, -1] > rel[0, 0]
    # Métrica: maior valor = mais longe -> mais escuro.
    met = normalize_for_display(depth_map, DepthOutputType.METRIC_DEPTH)
    assert met[-1, -1] < met[0, 0]
    assert 0.0 <= met.min() and met.max() <= 1.0


def test_constant_and_nan_maps_are_safe() -> None:
    const = np.full((5, 5), 3.0, dtype=np.float32)
    assert np.isfinite(normalize_for_display(const, "relative_inverse_depth")).all()
    with_nan = const.copy()
    with_nan[0, 0] = np.nan
    vmin, vmax = display_range(with_nan)
    assert np.isfinite([vmin, vmax]).all()


def test_image_conversions(depth_map) -> None:
    gray = to_grayscale_image(depth_map, "relative_inverse_depth")
    color = to_color_image(depth_map, "metric_depth")
    assert gray.shape == (10, 20) and gray.dtype == np.uint8
    assert color.shape == (10, 20, 3) and color.dtype == np.uint8


def test_plot_comparison_and_png_export(tmp_path, dummy_estimator, rgb_image) -> None:
    result = dummy_estimator.predict(rgb_image)
    fig = plot_comparison(rgb_image, [result, result], title="teste")
    assert len(fig.axes) >= 6  # 2 linhas x 3 colunas (+ barras de cor)
    plt.close(fig)
    paths = save_depth_visualizations(result, tmp_path)
    assert all(p.is_file() for p in paths.values())
