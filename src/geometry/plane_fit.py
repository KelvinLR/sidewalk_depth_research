"""Ajuste de plano dominante e remoção da rampa de perspectiva (Detrending).

Metodologia:
Em fotografias de calçada com ângulo oblíquo, a profundidade varia continuamente
de ~1 metro (primeiro plano) até 10–15 metros (fundo). Essa inclinação global ("rampa
de perspectiva") domina a escala de valores e obscurece variações locais milimétricas
ou centimétricas (buracos, desníveis de pavers, degraus).

Duas abordagens complementares são oferecidas:
1. **Ajuste 2D no espaço de imagem (Disparidade / Inverso de Profundidade)**:
   Em projeção perspectiva pinhole, a disparidade 1/Z de um plano 3D (o piso da calçada)
   é ESTRITAMENTE uma função afim em coordenadas de imagem:
       d(u, v) = a * u + b * v + c
   Ajustar esse plano afim via RANSAC elimina os pixels de fundo/paredes e os próprios
   defeitos (outliers), permitindo subtrair a rampa global com custo computacional mínimo.

2. **Ajuste 3D na Nuvem de Pontos Retroprojetada (Profundidade Métrica)**:
   Para modelos métricos (com profundidade Z em metros), os pixels são retroprojetados
   em coordenadas 3D (X, Y, Z) usando os parâmetros intrínsecos da câmera. O plano
   dominante A*X + B*Y + C*Z + D = 0 é ajustado via RANSAC 3D e a distância assinada
   ponto-plano fornece o desnível de elevação em metros (Δh).

Convenção de sinal:
- **Negativo (< 0)**: Depressão / Buraco (ponto mais fundo que o plano de referência).
- **Zero (~ 0)**: Superfície de referência da calçada.
- **Positivo (> 0)**: Elevação / Obstáculo / Degrau saliente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.inference.base_estimator import DepthOutputType


@dataclass
class PlaneFitResult:
    """Resultado do ajuste do plano dominante do piso da calçada.

    Attributes:
        residual: Mapa 2D (H, W) de desvio assinado em relação ao plano.
                  Convenção unificada: negativo = depressão/buraco,
                  positivo = elevação/degrau.
        plane: Mapa 2D (H, W) da superfície planar ajustada de referência.
        coefficients: Coeficientes do plano ([a, b, c] para 2D ou [A, B, C, D] para 3D).
        inlier_mask: Máscara booleana (H, W) indicando os pixels pertencentes ao piso.
        inlier_ratio: Proporção de inliers na região analisada (0.0 a 1.0).
        rmse: Erro quadrático médio dos inliers em relação ao plano.
        method: Nome do método utilizado ('ransac_2d', 'ransac_3d', 'lstsq').
    """

    residual: np.ndarray
    plane: np.ndarray
    coefficients: np.ndarray
    inlier_mask: np.ndarray
    inlier_ratio: float
    rmse: float
    method: str


def fit_ground_plane_2d(
    depth: np.ndarray,
    output_type: DepthOutputType | str,
    mask: np.ndarray | None = None,
    distance_threshold: float | None = None,
    max_iterations: int = 300,
    random_state: int | None = 42,
) -> PlaneFitResult:
    """Ajusta o plano dominante do piso diretamente sobre o mapa 2D via RANSAC.

    Em projeção pinhole, o inverso de profundidade (disparidade) de um plano 3D
    corresponde exatamente a um plano afim no plano da imagem:
        P(u, v) = a * u + b * v + c

    Args:
        depth: Array 2D float32 com o mapa de profundidade ou disparidade.
        output_type: Tipo de saída (DepthOutputType.RELATIVE_INVERSE_DEPTH ou METRIC_DEPTH).
        mask: Máscara booleana opcional (H, W) delimitando a região de calçada para
              o ajuste (True = analisar). Se None, usa os 70% inferiores da imagem.
        distance_threshold: Limiar de resíduo para classificar um pixel como inlier.
                            Se None, usa 3% do desvio interquartil (IQR).
        max_iterations: Número de iterações do RANSAC.
        random_state: Semente para reprodutibilidade.

    Returns:
        PlaneFitResult contendo o mapa de resíduo despolarizado, plano ajustado e métricas.
    """
    output_type = DepthOutputType(output_type)
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth deve ser 2D, recebido formato {depth_arr.shape}")

    h, w = depth_arr.shape
    yy, xx = np.mgrid[:h, :w]

    # Região de interesse padrão: 70% inferiores da imagem (onde a calçada predomina)
    if mask is None:
        roi = (yy >= int(h * 0.3)) & np.isfinite(depth_arr)
    else:
        roi = np.asarray(mask, dtype=bool) & np.isfinite(depth_arr)

    if not np.any(roi):
        raise ValueError("Máscara não contém pixels válidos para ajuste de plano.")

    x_pts = xx[roi].astype(np.float64)
    y_pts = yy[roi].astype(np.float64)
    z_pts = depth_arr[roi].astype(np.float64)
    n_pts = len(x_pts)

    if n_pts < 3:
        raise ValueError(f"Número insuficiente de pontos válidos ({n_pts}) para ajuste de plano.")

    # Definição do limiar de inlier
    if distance_threshold is None:
        q75, q25 = np.percentile(z_pts, [75, 25])
        iqr = float(q75 - q25)
        distance_threshold = max(float(iqr * 0.05), 1e-4)

    rng = np.random.default_rng(random_state)
    best_inlier_count = -1
    best_coeffs = np.zeros(3, dtype=np.float64)

    # Execução RANSAC
    for _ in range(max_iterations):
        sample_indices = rng.choice(n_pts, size=3, replace=False)
        pts_x = x_pts[sample_indices]
        pts_y = y_pts[sample_indices]
        pts_z = z_pts[sample_indices]

        # Matriz 3x3: [u, v, 1]
        a_mat = np.column_stack([pts_x, pts_y, np.ones(3, dtype=np.float64)])
        try:
            coeffs = np.linalg.solve(a_mat, pts_z)
        except np.linalg.LinAlgError:
            continue

        predicted = coeffs[0] * x_pts + coeffs[1] * y_pts + coeffs[2]
        errors = np.abs(z_pts - predicted)
        inlier_count = int(np.sum(errors < distance_threshold))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_coeffs = coeffs

    # Refinamento por mínimos quadrados com todos os inliers encontrados
    predicted_all = best_coeffs[0] * x_pts + best_coeffs[1] * y_pts + best_coeffs[2]
    inliers_subset = np.abs(z_pts - predicted_all) < distance_threshold

    if np.sum(inliers_subset) >= 3:
        design_matrix = np.column_stack([x_pts[inliers_subset], y_pts[inliers_subset], np.ones(np.sum(inliers_subset))])
        refined_coeffs, _, _, _ = np.linalg.lstsq(design_matrix, z_pts[inliers_subset], rcond=None)
    else:
        refined_coeffs = best_coeffs

    # Superfície planar extrapolada para toda a imagem
    full_plane = (refined_coeffs[0] * xx + refined_coeffs[1] * yy + refined_coeffs[2]).astype(np.float32)

    # Máscara completa de inliers
    full_inlier_mask = np.zeros((h, w), dtype=bool)
    inlier_errors = np.abs(depth_arr[roi] - full_plane[roi]) < distance_threshold
    full_inlier_mask[roi] = inlier_errors

    inlier_count_total = np.sum(full_inlier_mask[roi])
    inlier_ratio = float(inlier_count_total / n_pts)
    rmse = float(np.sqrt(np.mean((depth_arr[full_inlier_mask] - full_plane[full_inlier_mask]) ** 2))) if inlier_count_total > 0 else 0.0

    # Resíduo alinhado com a convenção:
    # Se relative_inverse_depth (maior = mais perto):
    #   Hole é mais longe -> depth < plane -> (depth - plane) < 0 (negativo)
    # Se metric_depth (maior = mais longe):
    #   Hole é mais longe -> depth > plane -> (plane - depth) < 0 (negativo)
    if output_type.larger_is_closer:
        residual = (depth_arr - full_plane).astype(np.float32)
    else:
        residual = (full_plane - depth_arr).astype(np.float32)

    return PlaneFitResult(
        residual=residual,
        plane=full_plane,
        coefficients=refined_coeffs.astype(np.float32),
        inlier_mask=full_inlier_mask,
        inlier_ratio=inlier_ratio,
        rmse=rmse,
        method="ransac_2d",
    )


def fit_ground_plane_3d(
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    distance_threshold: float = 0.05,
    max_iterations: int = 500,
    random_state: int | None = 42,
) -> PlaneFitResult:
    """Ajusta o plano 3D do piso na nuvem de pontos retroprojetada a partir de profundidade métrica.

    Requer saída métrica em metros. Converte (u, v, Z) em (X, Y, Z) e ajusta o plano:
        A * X + B * Y + C * Z + D = 0

    Args:
        depth: Array 2D float32 com profundidade métrica em metros (Z).
        mask: Máscara booleana opcional (H, W) indicando a região para ajuste do plano.
              Se None, usa os 70% inferiores da imagem.
        fx, fy: Distância focal em pixels. Se None, assume aproximação padrão fx = fy = largura da imagem.
        cx, cy: Ponto principal em pixels. Se None, assume centro da imagem (w/2, h/2).
        distance_threshold: Limiar de distância em metros para inliers no RANSAC (padrão 5 cm = 0.05 m).
        max_iterations: Iterações máximas do RANSAC.
        random_state: Semente aleatória.

    Returns:
        PlaneFitResult com o mapa de elevação Δh(u, v) em metros, plano 3D e métricas.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth deve ser 2D, recebido formato {depth_arr.shape}")

    h, w = depth_arr.shape
    fx = float(fx if fx is not None else w)
    fy = float(fy if fy is not None else w)
    cx = float(cx if cx is not None else w / 2.0)
    cy = float(cy if cy is not None else h / 2.0)

    yy, xx = np.mgrid[:h, :w]
    if mask is None:
        valid = (yy >= int(h * 0.3)) & np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < 50.0)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_arr) & (depth_arr > 0.05) & (depth_arr < 50.0)

    if not np.any(valid):
        raise ValueError("Nenhum ponto válido com profundidade métrica finita positiva.")

    # Retroprojeção para nuvem 3D (X, Y, Z) no referencial da câmera
    z_vals = depth_arr[valid].astype(np.float64)
    x_vals = (xx[valid].astype(np.float64) - cx) * z_vals / fx
    y_vals = (yy[valid].astype(np.float64) - cy) * z_vals / fy
    pts_3d = np.column_stack([x_vals, y_vals, z_vals])
    n_pts = len(pts_3d)

    if n_pts < 3:
        raise ValueError(f"Número insuficiente de pontos 3D ({n_pts}) para ajuste de plano.")

    rng = np.random.default_rng(random_state)
    best_inlier_count = -1
    best_plane = np.zeros(4, dtype=np.float64)

    # RANSAC 3D
    for _ in range(max_iterations):
        sample_indices = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = pts_3d[sample_indices]

        # Vetor normal n = (p2 - p1) x (p3 - p1)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-7:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        # Distância ortogonal assinada dos pontos ao plano
        distances = np.abs(np.dot(pts_3d, normal) + d)
        inlier_count = int(np.sum(distances < distance_threshold))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)

    # Refinamento do plano por SVD nos inliers
    normal = best_plane[:3]
    d = best_plane[3]
    inliers = np.abs(np.dot(pts_3d, normal) + d) < distance_threshold

    if np.sum(inliers) >= 3:
        pts_inliers = pts_3d[inliers]
        centroid = np.mean(pts_inliers, axis=0)
        centered = pts_inliers - centroid
        _, _, vh = np.linalg.svd(centered)
        normal_refined = vh[2, :]  # Autovetor associado ao menor autovalor
        normal_refined /= np.linalg.norm(normal_refined)
        # Garante que a normal aponte para cima (Y aponta para baixo na câmera)
        if normal_refined[1] > 0:
            normal_refined = -normal_refined
        d_refined = -np.dot(normal_refined, centroid)
        best_plane = np.array([normal_refined[0], normal_refined[1], normal_refined[2], d_refined], dtype=np.float64)

    a, b, c, d_coeff = best_plane
    # Para cada raio (u, v): X = (u - cx)*Z/fx, Y = (v - cy)*Z/fy
    # A*(u-cx)/fx * Z + B*(v-cy)/fy * Z + C * Z + D = 0
    # Z_plane(u, v) = -D / (A*(u-cx)/fx + B*(v-cy)/fy + C)
    denom = a * (xx.astype(np.float64) - cx) / fx + b * (yy.astype(np.float64) - cy) / fy + c
    # Evita divisão por zero
    denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    z_plane_full = -d_coeff / denom_safe

    # Máscara de inliers em 2D
    full_inlier_mask = np.zeros((h, w), dtype=bool)
    pts_dist = np.abs(np.dot(pts_3d, best_plane[:3]) + best_plane[3])
    full_inlier_mask[valid] = pts_dist < distance_threshold

    inlier_ratio = float(np.sum(full_inlier_mask[valid]) / n_pts)
    rmse = float(np.sqrt(np.mean(pts_dist[inliers] ** 2))) if np.sum(inliers) > 0 else 0.0

    # Resíduo métrico em metros:
    # Depressão (buraco) -> ponto está mais longe da câmera que o piso -> Z > Z_plane
    # Para convenção negativo = depressão: Δh = Z_plane - Z
    residual = (z_plane_full - depth_arr).astype(np.float32)
    residual[~valid] = 0.0

    return PlaneFitResult(
        residual=residual,
        plane=z_plane_full.astype(np.float32),
        coefficients=best_plane.astype(np.float32),
        inlier_mask=full_inlier_mask,
        inlier_ratio=inlier_ratio,
        rmse=rmse,
        method="ransac_3d",
    )
