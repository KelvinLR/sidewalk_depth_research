"""Persistência dos resultados de inferência (dados, não visualização).

Para cada par (imagem, modelo) são gravados em
``outputs/depth_maps/<imagem>/``:

- ``<model_key>.npz`` — arrays ``float32`` SEM normalização:
  ``depth`` (resolução original) e ``raw_output`` (resolução do modelo);
- ``<model_key>.json`` — metadados (modelo, natureza da saída, unidade, tempos,
  configuração, imagem de origem e seu SHA-256).

Além disso, cada execução é anexada a ``outputs/reports/inference_log.csv``,
formando um histórico dos experimentos.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, get_path
from src.inference.base_estimator import DepthOutputType, DepthResult

LOG_COLUMNS = [
    "created_at",
    "image_path",
    "image_sha256",
    "image_height",
    "image_width",
    "model_key",
    "model_name",
    "output_type",
    "units",
    "device",
    "model_input_shape",
    "raw_output_shape",
    "preprocess_time_s",
    "inference_time_s",
    "postprocess_time_s",
    "raw_min",
    "raw_max",
    "raw_median",
    "artifact",
]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 de um arquivo (identifica a imagem exata usada no experimento)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def result_dir_for(image_path: Path | str, base_dir: Path | None = None) -> Path:
    """Diretório de saída de uma imagem: ``outputs/depth_maps/<stem>``."""
    base = base_dir if base_dir is not None else get_path("depth_maps_dir", create=True)
    out = Path(base) / Path(image_path).stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_depth_result(
    result: DepthResult,
    image_path: Path | str,
    base_dir: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Salva os arrays brutos e os metadados de ``result``.

    Returns:
        Caminho do arquivo ``.npz`` gravado.
    """
    image_path = Path(image_path)
    out_dir = result_dir_for(image_path, base_dir)
    npz_path = out_dir / f"{result.model_key}.npz"
    np.savez_compressed(npz_path, depth=result.depth, raw_output=result.raw_output)

    metadata = result.to_metadata()
    metadata["image"] = {
        "path": _relative_to_root(image_path),
        "sha256": file_sha256(image_path) if image_path.is_file() else None,
    }
    metadata["arrays"] = {
        "file": npz_path.name,
        "depth": "resolução original (H, W), float32, interpolada da saída bruta, NÃO normalizada",
        "raw_output": "saída bruta do modelo (h, w), float32, NÃO normalizada",
    }
    if extra_metadata:
        metadata["extra"] = extra_metadata
    with (out_dir / f"{result.model_key}.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)
    return npz_path


def load_depth_result(npz_path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carrega arrays e metadados salvos por :func:`save_depth_result`."""
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        arrays = {k: data[k] for k in data.files}
    meta_path = npz_path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if "output_type" in metadata:
        metadata["output_type"] = DepthOutputType(metadata["output_type"])
    return arrays, metadata


def append_inference_log(
    result: DepthResult,
    image_path: Path | str,
    artifact: Path | str | None = None,
    log_path: Path | None = None,
) -> Path:
    """Anexa uma linha ao histórico ``outputs/reports/inference_log.csv``."""
    log_path = log_path if log_path is not None else get_path("reports_dir", create=True) / "inference_log.csv"
    image_path = Path(image_path)
    stats = result.stats()
    row = {
        "created_at": result.created_at,
        "image_path": _relative_to_root(image_path),
        "image_sha256": file_sha256(image_path) if image_path.is_file() else "",
        "image_height": result.image_size[0],
        "image_width": result.image_size[1],
        "model_key": result.model_key,
        "model_name": result.model_name,
        "output_type": result.output_type.value,
        "units": result.units or "",
        "device": result.device,
        "model_input_shape": "x".join(map(str, result.model_input_shape)),
        "raw_output_shape": "x".join(map(str, result.output_shape)),
        "preprocess_time_s": f"{result.preprocess_time_s:.6f}",
        "inference_time_s": f"{result.inference_time_s:.6f}",
        "postprocess_time_s": f"{result.postprocess_time_s:.6f}",
        "raw_min": f"{stats['min']:.6g}",
        "raw_max": f"{stats['max']:.6g}",
        "raw_median": f"{stats['median']:.6g}",
        "artifact": _relative_to_root(Path(artifact)) if artifact else "",
    }
    write_header = not log_path.is_file() or log_path.stat().st_size == 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return log_path
