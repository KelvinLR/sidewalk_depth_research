"""Visualização de mapas de profundidade (separada da inferência).

Princípios:

- **Nunca altera os dados**: as funções recebem os arrays brutos e produzem
  cópias normalizadas apenas para exibição/PNG. Os valores quantitativos ficam
  nos ``.npz`` salvos por ``src.inference.results_io``.
- **Convenção visual única "perto = claro"**: saídas relativas (disparidade,
  maior = mais perto) usam o colormap direto; saídas métricas (maior = mais
  longe) usam o colormap invertido. Assim os mapas são comparáveis visualmente,
  enquanto a barra de cores continua mostrando os valores reais (metros, para o
  modelo métrico; unidades arbitrárias, para os relativos).
- **Recorte por percentis** (padrão 2–98 %) apenas no contraste da exibição,
  para que poucos pixels extremos não "achatem" o restante do mapa.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.config import load_settings
from src.inference.base_estimator import DepthOutputType, DepthResult


def _viz_settings() -> dict:
    return load_settings().get("visualization", {})


def _percentiles(percentiles: tuple[float, float] | None) -> tuple[float, float]:
    if percentiles is None:
        p = _viz_settings().get("percentile_clip", [2.0, 98.0])
        percentiles = (float(p[0]), float(p[1]))
    lo, hi = percentiles
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError(f"Percentis inválidos: {percentiles}")
    return lo, hi


def display_range(depth: np.ndarray, percentiles: tuple[float, float] | None = None) -> tuple[float, float]:
    """Intervalo ``(vmin, vmax)`` para exibição, por percentis dos valores finitos."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = _percentiles(percentiles)
    vmin, vmax = np.percentile(finite, [lo, hi])
    if not vmax > vmin:  # mapa constante
        vmin, vmax = float(finite.min()), float(finite.min()) + 1e-6
    return float(vmin), float(vmax)


def normalize_for_display(
    depth: np.ndarray,
    output_type: DepthOutputType | str,
    percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Cópia normalizada em ``[0, 1]`` com convenção "perto = 1".

    Destinada SOMENTE à visualização. NaN/inf viram 0.
    """
    output_type = DepthOutputType(output_type)
    vmin, vmax = display_range(depth, percentiles)
    norm = (np.asarray(depth, dtype=np.float32) - vmin) / (vmax - vmin)
    norm = np.clip(np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if not output_type.larger_is_closer:
        norm = 1.0 - norm
    return norm.astype(np.float32, copy=False)


def _cmap_name(base: str, output_type: DepthOutputType) -> str:
    """Colormap com a convenção "perto = claro" (invertido para saídas métricas)."""
    if output_type.larger_is_closer:
        return base
    return base[:-2] if base.endswith("_r") else f"{base}_r"


def to_grayscale_image(
    depth: np.ndarray, output_type: DepthOutputType | str, percentiles: tuple[float, float] | None = None
) -> np.ndarray:
    """Mapa em escala de cinza ``uint8`` ``(H, W)`` (perto = claro)."""
    return np.round(normalize_for_display(depth, output_type, percentiles) * 255).astype(np.uint8)


def to_color_image(
    depth: np.ndarray,
    output_type: DepthOutputType | str,
    colormap: str | None = None,
    percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Mapa colorido RGB ``uint8`` ``(H, W, 3)`` (perto = claro no colormap)."""
    colormap = colormap or _viz_settings().get("colormap", "inferno")
    norm = normalize_for_display(depth, output_type, percentiles)
    rgba = matplotlib.colormaps[colormap](norm)
    return np.round(rgba[..., :3] * 255).astype(np.uint8)


def describe_result(result: DepthResult) -> str:
    """Rótulo curto: natureza da saída e unidade."""
    if result.is_metric:
        return f"métrica ({result.units})"
    return "relativa (inverso de profundidade, sem unidade)"


def plot_comparison(
    image_rgb: np.ndarray,
    results: Sequence[DepthResult],
    title: str | None = None,
    colormap: str | None = None,
    percentiles: tuple[float, float] | None = None,
    panel_width: float = 4.2,
) -> Figure:
    """Comparação lado a lado: uma linha por modelo, colunas RGB | cinza | colorido.

    Cada linha informa o nome do modelo, a natureza da saída e o tempo de
    inferência. A barra de cores mostra os valores REAIS da saída (com o
    recorte de percentis aplicado apenas ao contraste).
    """
    if not results:
        raise ValueError("Nenhum resultado para exibir.")
    colormap = colormap or _viz_settings().get("colormap", "inferno")
    h, w = image_rgb.shape[:2]
    panel_height = panel_width * h / w
    fig, axes = plt.subplots(
        len(results),
        3,
        figsize=(3 * panel_width + 2.5, len(results) * panel_height + 1.0),
        squeeze=False,
        dpi=_viz_settings().get("figure_dpi", 120),
    )
    plt.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.05, wspace=0.15, hspace=0.25)
    
    for row_idx, (row, result) in enumerate(zip(axes, results)):
        vmin, vmax = display_range(result.depth, percentiles)
        row[0].imshow(image_rgb)
        if row_idx == 0:
            row[0].set_title("RGB Original", fontsize=10, pad=6)

        row[1].imshow(to_grayscale_image(result.depth, result.output_type, percentiles), cmap="gray", vmin=0, vmax=255)
        if row_idx == 0:
            row[1].set_title("Escala de Cinza (perto = claro)", fontsize=10, pad=6)

        im = row[2].imshow(
            result.depth, cmap=_cmap_name(colormap, result.output_type), vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        if row_idx == 0:
            row[2].set_title(f"Colorido ({colormap})", fontsize=10, pad=6)
            
        cbar = fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.04)
        cbar.set_label("metros" if result.is_metric else "disparidade relativa", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Rótulo lateral claro e legível para cada modelo
        row_label = (
            f"{result.model_name}\n"
            f"Tipo: {result.output_type.value}\n"
            f"Inferência: {result.inference_time_s * 1000:.0f} ms ({result.device})"
        )
        row[0].set_ylabel(row_label, fontsize=8, labelpad=10, multialignment="center")

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    return fig


def save_depth_visualizations(
    result: DepthResult,
    out_dir: Path | str,
    colormap: str | None = None,
    percentiles: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Salva PNGs de visualização (``<model_key>_gray.png`` e ``<model_key>_color.png``).

    Os PNGs são apenas ilustrativos (8 bits, normalizados); use o ``.npz`` para análise.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gray": out_dir / f"{result.model_key}_gray.png",
        "color": out_dir / f"{result.model_key}_color.png",
    }
    gray = to_grayscale_image(result.depth, result.output_type, percentiles)
    color = to_color_image(result.depth, result.output_type, colormap, percentiles)
    if not cv2.imwrite(str(paths["gray"]), gray):
        raise OSError(f"Falha ao salvar {paths['gray']}")
    if not cv2.imwrite(str(paths["color"]), cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Falha ao salvar {paths['color']}")
    return paths


def save_figure(fig: Figure, path: Path | str, dpi: int | None = None) -> Path:
    """Salva uma figura (cria diretórios se necessário)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or _viz_settings().get("figure_dpi", 120), bbox_inches="tight")
    return path
