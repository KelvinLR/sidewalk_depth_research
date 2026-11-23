"""Testes básicos de estrutura, configuração e esquema do dataset.

Não baixam pesos nem executam modelos.
"""

from __future__ import annotations

import csv

import pytest

from src.config import (
    PROJECT_ROOT,
    ConfigError,
    get_model_config,
    get_path,
    load_settings,
    resolve_path,
)

EXPECTED_CSV_COLUMNS = [
    "image_name",
    "category",
    "location_id",
    "real_depth_cm",
    "real_height_difference_cm",
    "real_width_cm",
    "real_length_cm",
    "camera_height_cm",
    "camera_distance_cm",
    "camera_angle",
    "lighting_condition",
    "pavement_type",
    "notes",
]

EXPECTED_CATEGORIES = {"regular", "potholes", "uneven_surfaces", "ramps", "challenging_conditions"}
EXPECTED_MODELS = {
    "depth_anything_v2_small": "relative_inverse_depth",
    "depth_anything_v2_metric_outdoor_small": "metric_depth",
    "midas_v21_small": "relative_inverse_depth",
    "yolo26n_depth": "metric_depth",
}


def test_project_root_contains_settings() -> None:
    assert (PROJECT_ROOT / "config" / "settings.yaml").is_file()


def test_paths_are_independent_of_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert get_path("depth_maps_dir") == PROJECT_ROOT / "outputs" / "depth_maps"
    assert resolve_path("datasets").parent == PROJECT_ROOT


@pytest.mark.parametrize(
    "key", ["images_dir", "measurements_csv", "models_cache_dir", "depth_maps_dir", "comparisons_dir", "reports_dir"]
)
def test_all_configured_paths_resolve_inside_project(key: str) -> None:
    path = get_path(key)
    assert path.is_absolute()
    assert PROJECT_ROOT in path.parents


def test_unknown_path_key_raises() -> None:
    with pytest.raises(ConfigError):
        get_path("does_not_exist")


def test_dataset_category_folders_exist() -> None:
    categories = set(load_settings()["dataset"]["categories"])
    assert categories == EXPECTED_CATEGORIES
    images_dir = get_path("images_dir")
    for category in categories:
        assert (images_dir / category).is_dir(), f"Pasta ausente: {category}"


def test_output_folders_exist() -> None:
    for key in ("depth_maps_dir", "comparisons_dir", "reports_dir"):
        assert get_path(key).is_dir()


def test_measurements_csv_has_expected_header() -> None:
    with get_path("measurements_csv").open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == EXPECTED_CSV_COLUMNS


def test_measurements_csv_accepts_missing_values() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(get_path("measurements_csv"))
    assert list(df.columns) == EXPECTED_CSV_COLUMNS
    # Uma linha sem medições físicas deve ser lida como NaN, não como zero.
    row = {c: "" for c in EXPECTED_CSV_COLUMNS} | {"image_name": "x.jpg", "category": "regular"}
    df2 = pd.DataFrame([row]).replace("", float("nan"))
    assert df2["real_depth_cm"].isna().all()


@pytest.mark.parametrize("model_key,output_type", EXPECTED_MODELS.items())
def test_model_configs_declare_output_semantics(model_key: str, output_type: str) -> None:
    cfg = get_model_config(model_key)
    assert cfg["output_type"] == output_type
    assert cfg["backend"] in {"transformers", "torch_hub", "ultralytics"}
    if output_type == "metric_depth":
        assert cfg["units"] == "meters"
    else:
        assert cfg["units"] is None, "Saídas relativas não podem declarar unidade física."


def test_unknown_model_raises() -> None:
    with pytest.raises(ConfigError):
        get_model_config("not_a_model")
