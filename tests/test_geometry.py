"""Testes para ajuste de plano dominante e detrending (src.geometry)."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import (
    PlaneFitResult,
    fit_ground_plane_2d,
    fit_ground_plane_3d,
)
from src.inference import DepthOutputType


def test_fit_ground_plane_2d_disparity() -> None:
    # Cria uma rampa sintética afim: d(u, v) = 0.005 * v + 1.0 (onde v é linha y)
    h, w = 60, 80
    yy, xx = np.mgrid[:h, :w]
    depth = (0.005 * yy + 1.0).astype(np.float32)

    # Injeta um buraco (depressão): em disparidade, valor menor = mais longe
    depth[35:45, 35:45] -= 0.4

    res = fit_ground_plane_2d(
        depth,
        output_type=DepthOutputType.RELATIVE_INVERSE_DEPTH,
        distance_threshold=0.05,
        max_iterations=100,
        random_state=42,
    )

    assert isinstance(res, PlaneFitResult)
    assert res.method == "ransac_2d"
    assert res.inlier_ratio > 0.85
    # Na calçada (fora do buraco), o resíduo deve ser próximo de zero
    assert abs(res.residual[50, 10]) < 0.05
    # No buraco, o resíduo deve ser negativo
    assert res.residual[40, 40] < -0.3


def test_fit_ground_plane_2d_metric() -> None:
    # Rampa métrica Z(u, v) = 0.05 * (h - y) + 1.5
    h, w = 50, 50
    yy, xx = np.mgrid[:h, :w]
    depth = (0.05 * yy + 1.0).astype(np.float32)

    # Injeta um buraco (Z maior) e um degrau (Z menor)
    depth[30:35, 20:25] += 0.3  # buraco
    depth[40:45, 20:25] -= 0.3  # degrau elevado

    res = fit_ground_plane_2d(
        depth,
        output_type=DepthOutputType.METRIC_DEPTH,
        distance_threshold=0.05,
        max_iterations=100,
        random_state=42,
    )

    # Buraco deve ser negativo
    assert res.residual[32, 22] < -0.2
    # Degrau elevado deve ser positivo
    assert res.residual[42, 22] > 0.2


def test_fit_ground_plane_3d() -> None:
    # Plano 3D horizontal com piso Y = 1.2 m
    h, w = 60, 60
    yy, xx = np.mgrid[:h, :w]
    fx, fy = 60.0, 60.0
    cx, cy = 30.0, 30.0

    # Chão a Y = 1.2 m: Y = (v - cy)*Z/fy -> Z = 1.2 * fy / (v - cy) para v > cy (cy=30)
    depth = np.zeros((h, w), dtype=np.float32)
    valid_y = yy > 30
    depth[valid_y] = (1.2 * fy) / (yy[valid_y] - cy)
    depth[~valid_y] = np.nan

    # Injeta um buraco no chão
    depth[45:50, 25:30] += 0.4

    res = fit_ground_plane_3d(
        depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distance_threshold=0.08,
        max_iterations=150,
        random_state=42,
    )

    assert isinstance(res, PlaneFitResult)
    assert res.method == "ransac_3d"
    # No buraco, a elevação deve ser negativa (~ -0.4 m)
    assert res.residual[47, 27] < -0.2
    # Fora do buraco (chão plano), resíduo próximo de 0
    assert abs(res.residual[40, 20]) < 0.05


def test_geometry_validations() -> None:
    with pytest.raises(ValueError, match="depth deve ser 2D"):
        fit_ground_plane_2d(np.ones((5, 5, 3)), DepthOutputType.METRIC_DEPTH)

    with pytest.raises(ValueError, match="depth deve ser 2D"):
        fit_ground_plane_3d(np.ones((5, 5, 3)))

    with pytest.raises(ValueError, match="pixels válidos"):
        fit_ground_plane_2d(
            np.ones((10, 10)),
            DepthOutputType.METRIC_DEPTH,
            mask=np.zeros((10, 10), dtype=bool),
        )
