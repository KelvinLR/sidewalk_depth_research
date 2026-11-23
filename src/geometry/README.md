# `src/geometry/` — Análise Geométrica e Reconstrução 3D

Módulo dedicado à análise geométrica da calçada, estimativa de superfícies de referência
e remoção de rampas de perspectiva (detrending).

## Componentes implementados

### `plane_fit.py` — Ajuste de Plano Dominante e Detrending

| Função | Espaço | Entrada | Método | Saída |
|---|---|---|---|---|
| `fit_ground_plane_2d` | 2D / Imagem | Disparidade afim-invariante ou métrica | RANSAC em modelo afim $d(u, v) = a \cdot u + b \cdot v + c$ com refinamento por mínimos quadrados | `PlaneFitResult` com resíduo $\Delta h$, plano interpolado, máscara de inliers e RMSE |
| `fit_ground_plane_3d` | 3D / Câmera | Profundidade métrica ($Z$ em metros) + intrínsecos | Retroprojeção para nuvem 3D + RANSAC no plano $A X + B Y + C Z + D = 0$ com refinamento por SVD | `PlaneFitResult` com mapa de elevação métrica $\Delta h(u, v)$ em metros |

### Convenção unificada de sinal

Em todos os métodos deste módulo e da visualização:
- **Negativo ($< 0$)**: Depressão / Buraco (recesso abaixo do piso de referência).
- **Zero ($\approx 0$)**: Superfície plana regular da calçada.
- **Positivo ($> 0$)**: Elevação / Degrau / Obstáculo saliente.

## Restrições metodológicas

- **Profundidade relativa não é medida física**: O ajuste 2D em modelos relativos (Depth Anything V2 Small e MiDaS) elimina a rampa de perspectiva e isola defeitos com alto contraste visual, mas o resíduo permanece adimensional (não converte diretamente em centímetros).
- **Profundidade métrica**: Para modelos métricos (Depth Anything V2 Metric Outdoor e YOLO26n Depth), o `fit_ground_plane_3d` calcula a elevação física $\Delta h$ em metros.
