"""Visualização de mapas de profundidade (separada da inferência)."""

from src.visualization.depth_visualizer import (
    compute_depth_gradient,
    compute_local_relief,
    display_range,
    normalize_for_display,
    plot_batch_overview_grid,
    plot_comparison,
    plot_defect_inspection,
    plot_multi_model_relief_comparison,
    save_depth_visualizations,
    save_figure,
    to_color_image,
    to_grayscale_image,
)

__all__ = [
    "compute_depth_gradient",
    "compute_local_relief",
    "display_range",
    "normalize_for_display",
    "plot_batch_overview_grid",
    "plot_comparison",
    "plot_defect_inspection",
    "plot_multi_model_relief_comparison",
    "save_depth_visualizations",
    "save_figure",
    "to_color_image",
    "to_grayscale_image",
]
