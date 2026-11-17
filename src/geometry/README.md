# `src/geometry/` — análise geométrica (reservado)

> [!NOTE]
> Este módulo **ainda não está implementado**. Ele existe para receber, em etapas
> futuras, a lógica de reconstrução 3D e análise de superfície. Nada aqui deve ser
> usado para produzir medidas físicas na versão atual.

## Escopo previsto

| Tema | Descrição | Pré-requisitos |
|---|---|---|
| Calibração de câmera | Intrínsecos (fx, fy, cx, cy) e distorção; leitura de EXIF (distância focal) como aproximação | Padrão de calibração (xadrez) ou EXIF confiável |
| Nuvem de pontos | Retroprojeção `depth + intrínsecos → XYZ` (Open3D) | Profundidade **métrica** ou escala recuperada + intrínsecos |
| Estimativa de plano | Ajuste do plano dominante da calçada | Nuvem de pontos |
| RANSAC | Ajuste robusto de plano, tolerante a outliers (buracos, obstáculos) | Nuvem de pontos |
| Depressões / elevações | Distância assinada dos pontos ao plano de referência | Plano ajustado |
| Diferença de altura | Desnível entre placas / degraus | Segmentação de regiões + plano |
| Inclinação | Ângulo do plano em relação à vertical (gravidade) ou ao plano da câmera | Plano + orientação da câmera (IMU, futura integração) |

## Restrições metodológicas

- **Profundidade relativa não é medida física.** Saídas de Depth Anything V2 Small e
  MiDaS são disparidades relativas e afim-invariantes (escala e deslocamento
  desconhecidos). Não é possível obter centímetros diretamente delas.
- Para medir, será necessário pelo menos um dos itens abaixo, a ser investigado:
  - modelo métrico (ex.: Depth Anything V2 Metric Outdoor) **e** validação da sua
    precisão em curtas distâncias (calçada a 1–3 m, bem abaixo do domínio de treino
    em cenas de rua de até 80 m);
  - recuperação de escala/deslocamento com referência conhecida na cena
    (objeto de dimensão conhecida, altura da câmera, plano do chão);
  - calibração intrínseca da câmera.
- Toda função deste módulo deverá declarar explicitamente a unidade de entrada e de
  saída e recusar (ou sinalizar) entradas relativas quando exigir valores métricos.

## Convenções planejadas

- Entrada: arrays `float32` com a **saída bruta preservada** do modelo (nunca a versão
  normalizada para visualização), acompanhados do tipo de saída
  (`relative_inverse_depth` ou `metric_depth`).
- Sistema de coordenadas da câmera: X à direita, Y para baixo, Z para frente (OpenCV).
