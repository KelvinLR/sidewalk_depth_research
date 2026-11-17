# `src/evaluation/` — avaliação e validação (reservado)

> [!NOTE]
> Este módulo **ainda não está implementado**. Ele receberá, em etapas futuras, as
> rotinas que comparam estimativas com medições físicas de referência.

## Escopo previsto

1. **Leitura das referências** — `datasets/measurements.csv`, tratando campos de
   medição vazios como ausentes (`NaN`), nunca como zero.
2. **Associação imagem ↔ medição** — por `image_name` e `category`.
3. **Métricas de erro** (somente quando houver referência física e estimativa métrica):
   - erro absoluto e relativo (cm, %);
   - MAE / RMSE por categoria e por condição de captura;
   - análise de sensibilidade a `camera_height_cm`, `camera_distance_cm`,
     `camera_angle` e `lighting_condition`.
4. **Métricas de profundidade relativa** (sem unidade física) — ex.: AbsRel/δ após
   alinhamento por escala e deslocamento, quando houver mapa de profundidade de
   referência (datasets públicos com ground truth).
5. **Relatórios** — tabelas (Pandas) e figuras salvas em `outputs/reports/`.

## Restrições

- Não comparar valores relativos com medições em centímetros sem antes documentar o
  procedimento de recuperação de escala utilizado.
- Não preencher medições ausentes com valores estimados ou inventados.
