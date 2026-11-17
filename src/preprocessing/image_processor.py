"""Leitura de imagens RGB e utilitários de pré-processamento independentes de modelo.

O pré-processamento específico de cada rede (redimensionamento, normalização,
formato do tensor) fica no respectivo estimador. Aqui ficam apenas as etapas
comuns e neutras:

- leitura do arquivo;
- aplicação da orientação EXIF (fotos de celular costumam estar "deitadas" nos
  pixels, com a rotação apenas na tag EXIF);
- conversão para RGB ``uint8`` ``(H, W, 3)`` (remove alfa, converte tons de cinza/CMYK);
- extração de metadados EXIF úteis para calibração futura (câmera, distância focal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import get_path, load_settings, resolve_path

_EXIF_ORIENTATION_TAG = 0x0112
#: Tags EXIF relevantes para futura estimativa de intrínsecos da câmera.
_EXIF_TAGS_OF_INTEREST = {
    0x010F: "make",
    0x0110: "model",
    0x9003: "datetime_original",
    0x920A: "focal_length_mm",
    0xA405: "focal_length_35mm",
    0xA002: "exif_width",
    0xA003: "exif_height",
}


class ImageLoadError(RuntimeError):
    """Falha ao ler ou converter uma imagem."""


@dataclass(frozen=True)
class LoadedImage:
    """Imagem carregada e pronta para os estimadores.

    Attributes:
        path: caminho absoluto do arquivo.
        rgb: array ``uint8`` ``(H, W, 3)`` em RGB, já com a orientação EXIF aplicada.
        original_mode: modo PIL original (ex.: ``RGB``, ``RGBA``, ``L``).
        exif_orientation: valor da tag EXIF Orientation (1–8) ou ``None``.
        orientation_applied: se a rotação EXIF foi aplicada aos pixels.
        category: categoria do dataset (nome da pasta), se a imagem estiver em
            ``datasets/images/<categoria>/``.
        exif: metadados EXIF selecionados (câmera, distância focal...).
    """

    path: Path
    rgb: np.ndarray
    original_mode: str
    exif_orientation: int | None
    orientation_applied: bool
    category: str | None
    exif: dict[str, Any] = field(default_factory=dict)

    @property
    def size_hw(self) -> tuple[int, int]:
        return int(self.rgb.shape[0]), int(self.rgb.shape[1])

    @property
    def name(self) -> str:
        return self.path.name

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "category": self.category,
            "size_hw": list(self.size_hw),
            "original_mode": self.original_mode,
            "exif_orientation": self.exif_orientation,
            "orientation_applied": self.orientation_applied,
            "exif": self.exif,
        }


def _exif_value(value: Any) -> Any:
    """Converte valores EXIF (IFDRational, bytes) em tipos serializáveis."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator") and not isinstance(value, int):
            return float(value)
    except (ZeroDivisionError, ValueError):
        return None
    if isinstance(value, str):
        return value.strip("\x00 ")
    return value


def _extract_exif(img: Image.Image) -> tuple[int | None, dict[str, Any]]:
    exif = img.getexif()
    if not exif:
        return None, {}
    merged: dict[int, Any] = dict(exif)
    try:
        merged.update(exif.get_ifd(0x8769))  # Exif SubIFD (distância focal etc.)
    except (KeyError, AttributeError):
        pass
    orientation = merged.get(_EXIF_ORIENTATION_TAG)
    selected = {name: _exif_value(merged[tag]) for tag, name in _EXIF_TAGS_OF_INTEREST.items() if tag in merged}
    return (int(orientation) if orientation else None), selected


def _infer_category(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(get_path("images_dir"))
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def load_rgb_image(path: str | Path, apply_exif_orientation: bool | None = None) -> LoadedImage:
    """Carrega uma imagem como RGB ``uint8`` ``(H, W, 3)``.

    Args:
        path: caminho absoluto ou relativo à raiz do projeto.
        apply_exif_orientation: aplica a rotação EXIF. Padrão:
            ``preprocessing.apply_exif_orientation`` do settings.yaml.

    Raises:
        ImageLoadError: arquivo inexistente ou formato não suportado.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise ImageLoadError(f"Imagem não encontrada: {resolved}")
    if apply_exif_orientation is None:
        apply_exif_orientation = bool(load_settings().get("preprocessing", {}).get("apply_exif_orientation", True))

    try:
        with Image.open(resolved) as img:
            img.load()
            original_mode = img.mode
            orientation, exif = _extract_exif(img)
            applied = False
            if apply_exif_orientation and orientation not in (None, 1):
                img = ImageOps.exif_transpose(img)
                applied = True
            has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
            if has_alpha:
                # Compõe sobre fundo branco em vez de descartar o alfa (evita pixels "fantasma").
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                img = background
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    except UnidentifiedImageError as exc:
        hint = " (HEIC/HEIF não é suportado; converta para JPEG/PNG)" if resolved.suffix.lower() in {".heic", ".heif"} else ""
        raise ImageLoadError(f"Formato de imagem não reconhecido: {resolved}{hint}") from exc
    except OSError as exc:
        raise ImageLoadError(f"Falha ao ler {resolved}: {exc}") from exc

    return LoadedImage(
        path=resolved,
        rgb=rgb,
        original_mode=original_mode,
        exif_orientation=orientation,
        orientation_applied=applied,
        category=_infer_category(resolved),
        exif=exif,
    )


def list_dataset_images(category: str | None = None) -> list[Path]:
    """Lista as imagens em ``datasets/images/`` (opcionalmente de uma categoria).

    Returns:
        Caminhos absolutos ordenados por categoria e nome.
    """
    settings = load_settings()
    images_dir = get_path("images_dir")
    extensions = {e.lower() for e in settings.get("dataset", {}).get("image_extensions", [])}
    categories = settings.get("dataset", {}).get("categories", [])
    if category is not None and category not in categories:
        raise ValueError(f"Categoria desconhecida: {category!r}. Opções: {categories}")
    roots = [images_dir / category] if category else [images_dir / c for c in categories]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    return sorted(found)
