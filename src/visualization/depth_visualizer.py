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


def compute_local_relief(
    depth: np.ndarray,
    output_type: DepthOutputType | str = DepthOutputType.RELATIVE_INVERSE_DEPTH,
    sigma: float = 25.0,
) -> np.ndarray:
    """Calcula o mapa de relevo residual local (filtro passa-alta).

    Subtrai a tendência suave de baixa frequência (rampa de perspectiva da calçada)
    isolando descontinuidades geométricas locais (buracos, remendos, desníveis).

    Convenção de sinal unificada:
    - Negativo (< 0): Depressão / Buraco
    - Zero (~ 0): Superfície plana de referência
    - Positivo (> 0): Elevação / Degrau / Obstáculo

    Args:
        depth: Array 2D float32 com o mapa de profundidade ou disparidade.
        output_type: Tipo de saída (DepthOutputType.RELATIVE_INVERSE_DEPTH ou METRIC_DEPTH).
        sigma: Desvio padrão do filtro Gaussiano em pixels (padrão 25.0).
               Controla a escala espacial da rampa removida.

    Returns:
        Array 2D float32 com o relevo residual.
    """
    import cv2

    output_type = DepthOutputType(output_type)
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"depth deve ser 2D, recebido formato {arr.shape}")
    if sigma <= 0:
        raise ValueError(f"sigma deve ser positivo, recebido {sigma}")

    smooth = cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))

    if output_type.larger_is_closer:
        relief = arr - smooth
    else:
        relief = smooth - arr

    return relief.astype(np.float32, copy=False)


def compute_depth_gradient(depth: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Calcula a magnitude do gradiente espacial ||∇Z|| via operador Sobel.

    Destaca bordas e transições abruptas de profundidade (perímetro de buracos,
    juntas de pavers desencaixados, bordas de degraus).

    Args:
        depth: Array 2D float32 com mapa de profundidade ou disparidade.
        ksize: Tamanho do kernel Sobel (1, 3, 5 ou 7). Padrão 3.

    Returns:
        Array 2D float32 com a magnitude do gradiente espacial.
    """
    import cv2

    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"depth deve ser 2D, recebido formato {arr.shape}")
    if ksize not in (1, 3, 5, 7):
        raise ValueError(f"ksize do Sobel deve ser 1, 3, 5 ou 7, recebido {ksize}")

    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=ksize)
    grad = np.sqrt(gx**2 + gy**2)
    return grad.astype(np.float32, copy=False)


def plot_defect_inspection(
    image_rgb: np.ndarray,
    result: DepthResult,
    sigma: float = 25.0,
    title: str | None = None,
    colormap_depth: str | None = None,
    colormap_relief: str = "seismic",
    colormap_grad: str = "magma",
    percentile_clip: float = 98.0,
    panel_width: float = 4.2,
) -> Figure:
    """Painel de inspeção de defeito em 4 vistas: RGB | Profundidade | Relevo Residual | Gradiente.

    Remove a rampa de perspectiva no painel de relevo (divergente centrado em 0),
    evidenciando buracos e degraus com alto contraste visual.

    Args:
        image_rgb: Imagem RGB original uint8 (H, W, 3).
        result: DepthResult da inferência.
        sigma: Escala espacial para o filtro de relevo local.
        title: Título geral da figura.
        colormap_depth: Colormap para a profundidade global (padrão 'inferno').
        colormap_relief: Colormap divergente para o relevo residual (padrão 'seismic').
        colormap_grad: Colormap para o gradiente de borda (padrão 'magma').
        percentile_clip: Percentil para recorte simétrico do relevo e contraste do gradiente.
        panel_width: Largura de cada painel em polegadas.

    Returns:
        Matplotlib Figure configurada.
    """
    colormap_depth = colormap_depth or _viz_settings().get("colormap", "inferno")
    h, w = image_rgb.shape[:2]
    panel_height = panel_width * h / w

    relief = compute_local_relief(result.depth, result.output_type, sigma=sigma)
    grad = compute_depth_gradient(result.depth)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(4 * panel_width + 2.0, panel_height + 1.2),
        squeeze=True,
        dpi=_viz_settings().get("figure_dpi", 120),
    )
    plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08, wspace=0.25)

    # 1. RGB Original
    axes[0].imshow(image_rgb)
    axes[0].set_title("1. RGB Original", fontsize=10, pad=6)

    # 2. Profundidade Global
    vmin, vmax = display_range(result.depth)
    im_depth = axes[1].imshow(
        result.depth,
        cmap=_cmap_name(colormap_depth, result.output_type),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_title(f"2. Profundidade Bruta\n({result.model_name})", fontsize=10, pad=6)
    cbar_depth = fig.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_depth.set_label("metros" if result.is_metric else "disparidade", fontsize=8)
    cbar_depth.ax.tick_params(labelsize=7)

    # 3. Relevo Residual Local (Passa-Alta: Z - Z_smooth)
    finite_relief = relief[np.isfinite(relief)]
    lim = float(np.percentile(np.abs(finite_relief), percentile_clip)) if finite_relief.size > 0 else 1.0
    if lim <= 1e-6:
        lim = 1.0
    im_relief = axes[2].imshow(
        relief,
        cmap=colormap_relief,
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    unit_label = " (m)" if result.is_metric else ""
    axes[2].set_title(
        f"3. Relevo Residual Local (σ={sigma:.0f}px)\nAzul: Buraco | Vermelho: Elevação",
        fontsize=9,
        pad=6,
    )
    cbar_relief = fig.colorbar(im_relief, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_relief.set_label(f"desvio residual{unit_label}", fontsize=8)
    cbar_relief.ax.tick_params(labelsize=7)

    # 4. Gradiente de Borda ||∇Z||
    finite_grad = grad[np.isfinite(grad)]
    grad_max = float(np.percentile(finite_grad, percentile_clip)) if finite_grad.size > 0 else 1.0
    if grad_max <= 1e-6:
        grad_max = 1.0
    im_grad = axes[3].imshow(
        grad,
        cmap=colormap_grad,
        vmin=0,
        vmax=grad_max,
        interpolation="nearest",
    )
    axes[3].set_title("4. Gradiente Espacial ||∇Z||\n(Bordas e Desníveis)", fontsize=10, pad=6)
    cbar_grad = fig.colorbar(im_grad, ax=axes[3], fraction=0.046, pad=0.04)
    cbar_grad.set_label("magnitude da borda", fontsize=8)
    cbar_grad.ax.tick_params(labelsize=7)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    return fig


def plot_multi_model_relief_comparison(
    image_rgb: np.ndarray,
    results: Sequence[DepthResult],
    sigma: float = 25.0,
    title: str | None = None,
    colormap_relief: str = "seismic",
    percentile_clip: float = 98.0,
    panel_width: float = 3.6,
) -> Figure:
    """Compara o relevo residual de múltiplos modelos lado a lado com a imagem RGB.

    Permite constatar visualmente qual modelo preserva as depressões e bordas
    de defeitos após a remoção da rampa de perspectiva.
    """
    if not results:
        raise ValueError("Nenhum resultado para exibir.")
    h, w = image_rgb.shape[:2]
    panel_height = panel_width * h / w
    n_cols = len(results) + 1

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(n_cols * panel_width + 1.5, panel_height + 1.0),
        squeeze=False,
        dpi=_viz_settings().get("figure_dpi", 120),
    )
    plt.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.08, wspace=0.20)
    ax_row = axes[0]

    # RGB
    ax_row[0].imshow(image_rgb)
    ax_row[0].set_title("RGB Original", fontsize=9, pad=6)
    ax_row[0].set_xticks([])
    ax_row[0].set_yticks([])

    for i, res in enumerate(results, start=1):
        relief = compute_local_relief(res.depth, res.output_type, sigma=sigma)
        finite = relief[np.isfinite(relief)]
        lim = float(np.percentile(np.abs(finite), percentile_clip)) if finite.size > 0 else 1.0
        if lim <= 1e-6:
            lim = 1.0
        im = ax_row[i].imshow(relief, cmap=colormap_relief, vmin=-lim, vmax=lim, interpolation="nearest")
        sub_title = f"{res.model_name}\n({res.output_type.value})"
        ax_row[i].set_title(sub_title, fontsize=8, pad=6)
        cbar = fig.colorbar(im, ax=ax_row[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        ax_row[i].set_xticks([])
        ax_row[i].set_yticks([])

    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    return fig


def plot_batch_overview_grid(
    images: Sequence[LoadedImage],
    results_by_image: dict[str, Sequence[DepthResult]],
    primary_relative_key: str = "depth_anything_v2_small",
    primary_metric_key: str = "yolo26n_depth",
    sigma: float = 25.0,
    title: str = "Visão Geral Consolidada de Todo o Lote — Calçadas do Dataset",
) -> Figure:
    """Gera uma matriz comparativa única com todas as imagens do lote em uma só figura.

    Cada linha representa uma imagem de calçada, com 5 colunas:
    1. RGB Original
    2. Estimador Relativo (Depth Anything V2 Small)
    3. Estimador Métrico (YOLO26n Depth)
    4. Relevo Residual Local (Filtro Passa-Alta: Azul = Buraco | Vermelho = Degrau)
    5. Gradiente Espacial de Borda ||∇Z||

    Args:
        images: Lista de LoadedImage contendo as imagens de entrada.
        results_by_image: Dicionário {nome_da_imagem: [DepthResult, ...]}.
        primary_relative_key: Chave do modelo relativo principal.
        primary_metric_key: Chave do modelo métrico principal.
        sigma: Escala espacial para o filtro de relevo residual.
        title: Título geral da figura.

    Returns:
        Matplotlib Figure contendo a matriz consolidada de todo o lote.
    """
    n_imgs = len(images)
    if n_imgs == 0:
        raise ValueError("A lista de imagens está vazia.")

    fig, axes = plt.subplots(n_imgs, 5, figsize=(20, 3.8 * n_imgs), squeeze=False, dpi=_viz_settings().get("figure_dpi", 120))

    for row_idx, img in enumerate(images):
        img_results = results_by_image.get(img.name, [])
        res_rel = next((r for r in img_results if r.model_key == primary_relative_key), img_results[0] if img_results else None)
        res_met = next((r for r in img_results if r.model_key == primary_metric_key), None)

        # 1. RGB Original
        axes[row_idx, 0].imshow(img.rgb)
        axes[row_idx, 0].set_ylabel(f"{img.name}\n({img.size_hw[1]}x{img.size_hw[0]} px)", fontsize=9, labelpad=8)
        if row_idx == 0:
            axes[row_idx, 0].set_title("1. RGB Original", fontsize=10, pad=6)

        # 2. Modelo Relativo
        if res_rel is not None:
            vmin, vmax = display_range(res_rel.depth)
            axes[row_idx, 1].imshow(res_rel.depth, cmap="inferno", vmin=vmin, vmax=vmax)
            if row_idx == 0:
                axes[row_idx, 1].set_title(f"2. {res_rel.model_name}\n(Disparidade Relativa)", fontsize=10, pad=6)

        # 3. Modelo Métrico
        if res_met is not None:
            vmin_m, vmax_m = display_range(res_met.depth)
            axes[row_idx, 2].imshow(res_met.depth, cmap="inferno_r", vmin=vmin_m, vmax=vmax_m)
            if row_idx == 0:
                axes[row_idx, 2].set_title(f"3. {res_met.model_name}\n(Profundidade Métrica)", fontsize=10, pad=6)
        elif res_rel is not None:
            axes[row_idx, 2].axis("off")

        # 4. Relevo Residual Local
        if res_rel is not None:
            relief = compute_local_relief(res_rel.depth, res_rel.output_type, sigma=sigma)
            finite_r = relief[np.isfinite(relief)]
            lim = float(np.percentile(np.abs(finite_r), 98)) if finite_r.size > 0 else 1.0
            if lim <= 1e-4:
                lim = 1.0
            axes[row_idx, 3].imshow(relief, cmap="seismic", vmin=-lim, vmax=lim)
            if row_idx == 0:
                axes[row_idx, 3].set_title(f"4. Relevo Residual (σ={sigma:.0f}px)\nAzul: Buraco | Vermelho: Degrau", fontsize=10, pad=6)

        # 5. Gradiente de Borda
        if res_rel is not None:
            grad = compute_depth_gradient(res_rel.depth)
            finite_g = grad[np.isfinite(grad)]
            gmax = float(np.percentile(finite_g, 98)) if finite_g.size > 0 else 1.0
            if gmax <= 1e-4:
                gmax = 1.0
            axes[row_idx, 4].imshow(grad, cmap="magma", vmin=0, vmax=gmax)
            if row_idx == 0:
                axes[row_idx, 4].set_title("5. Gradiente Espacial ||∇Z||\n(Bordas do Defeito)", fontsize=10, pad=6)

        for c in range(5):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    if title:
        fig.suptitle(title, fontsize=13, y=0.99)
    plt.tight_layout()
    return fig
