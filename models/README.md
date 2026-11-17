# `models/` — pesos pré-treinados

Os pesos **não são versionados no Git**. Eles são baixados automaticamente na
primeira execução e armazenados em `models/cache/` (ignorado pelo `.gitignore`).
Execuções seguintes reutilizam o cache, sem novo download.

```text
models/
├── README.md
└── cache/                       # criado automaticamente (não versionado)
    ├── huggingface/             # HF_HOME  -> Depth Anything V2 (Small e Metric Outdoor)
    └── torch/hub/               # torch.hub -> código do MiDaS + checkpoints
        ├── intel-isl_MiDaS_v3_1/
        ├── rwightman_gen-efficientnet-pytorch_master/
        └── checkpoints/
            ├── midas_v21_small_256.pt
            └── tf_efficientnet_lite3-b733e338.pth
```

O caminho do cache é definido em `config/settings.yaml` (`paths.models_cache_dir`).

---

## Investigação de compatibilidade (2026-11-16)

Os três modelos foram **carregados e executados com sucesso** no ambiente do
`requirements.txt` (Python 3.11.15, torch 2.15.0, transformers 5.23.0, timm 1.0.31,
macOS arm64, **CPU**). O teste usou uma **imagem sintética 640×480** apenas para
verificar carregamento, formatos de tensor e tipo de saída.

> [!IMPORTANT]
> Isso **não** valida a qualidade da profundidade em calçadas. A validação com
> fotografias reais é a Etapa 4. Os valores numéricos abaixo vieram de uma imagem
> sintética e não têm significado físico.

| | Depth Anything V2 Small | Depth Anything V2 Metric Outdoor Small | MiDaS v2.1 Small |
|---|---|---|---|
| Origem | Hugging Face `depth-anything/Depth-Anything-V2-Small-hf` | Hugging Face `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | `torch.hub` `intel-isl/MiDaS:v3_1` → `MiDaS_small` |
| Revisão testada | `5426e4f0f365…` | `fd2c22027eaf…` | tag `v3_1` + checkpoint `v2_1/midas_v21_small_256.pt` |
| Carregamento | `AutoImageProcessor` + `AutoModelForDepthEstimation` | idem | `torch.hub.load(...)` + `transforms.small_transform` |
| Arquitetura | DINOv2 ViT-S/14 + cabeça DPT (~25M params) | idem, cabeça métrica | EfficientNet-Lite3 + decoder MiDaS |
| Pré-processamento | `DPTImageProcessor`: lado menor → 518, mantém proporção, múltiplo de 14, bicúbico, normalização ImageNet | idêntico ao Small | Entrada `HxWx3` RGB `uint8` → `/255`, lado maior → 256, mantém proporção, múltiplo de 32, bicúbico (OpenCV), normalização ImageNet |
| Entrada 640×480 → tensor | `(1, 3, 518, 686)` | `(1, 3, 518, 686)` | `(1, 3, 192, 256)` |
| Saída bruta | `(1, 518, 686)` | `(1, 518, 686)` | `(1, 192, 256)` |
| Volta à resolução original | `processor.post_process_depth_estimation(..., target_sizes=[(H, W)])` | idem | interpolação manual (`torch.nn.functional.interpolate`, bicúbica) |
| `config.depth_estimation_type` | `relative` | `metric` (`max_depth = 80`) | — |
| **Natureza da saída** | **Disparidade relativa** (inverso de profundidade, afim-invariante). Maior = mais perto. Sem unidade. | **Profundidade métrica em metros**. Maior = mais longe. | **Disparidade relativa** (inverso de profundidade, afim-invariante). Maior = mais perto. Sem unidade. Escala numérica arbitrária (centenas). |
| Download | ~99 MB (safetensors) | ~99 MB (safetensors) | 96 MB (MiDaS) + 32 MB (EfficientNet-Lite3 ImageNet, baixado pelo construtor e depois sobrescrito) |
| Tempo 1ª inferência (CPU, M1, sintética) | ~0,9 s | ~0,7 s | ~0,4 s |
| Licença | Apache-2.0 | **Não declarada no model card** (o repositório oficial aplica Apache-2.0 às variantes Small; confirmar antes de uso comercial) | MIT |

### Observações importantes

1. **Prompt interativo do MiDaS (bloqueante).** O `MiDaS_small` chama internamente
   `torch.hub.load("rwightman/gen-efficientnet-pytorch", ...)` sem `trust_repo`. Em
   ambiente novo isso exibe `Do you trust this repository ... (y/N)?` e **trava o
   notebook**. Solução adotada (a implementar no carregador, Parte 5): pré-autorizar
   com `torch.hub.list("rwightman/gen-efficientnet-pytorch", trust_repo=True)` antes
   de carregar o MiDaS.
2. **Dependências implícitas do MiDaS.** O `hubconf.py` importa `timm` (backbones) e
   `torchvision` (transforms), mesmo para o modelo Small. Com `timm 1.0.31` funciona,
   mas emite `FutureWarning` sobre `timm.models.layers`. Versões futuras do `timm`
   podem quebrar o MiDaS — por isso a versão está fixada.
3. **Reprodutibilidade parcial do MiDaS.** O repositório do MiDaS está fixado na tag
   `v3_1`, mas o `gen-efficientnet-pytorch` é carregado do branch `master` (não
   fixável via hubconf do MiDaS). O repositório está arquivado, o que reduz o risco
   de mudanças.
4. **Metric Outdoor fora do domínio.** O modelo métrico foi ajustado no Virtual KITTI 2
   (cenas de rua sintéticas, até 80 m, câmera de veículo). Fotos de calçada a 1–3 m,
   com câmera inclinada para baixo, estão fora desse domínio; os valores em metros
   **precisam ser validados** antes de qualquer uso quantitativo.
5. **Primeira execução exige internet.** Depois disso, os modelos funcionam offline a
   partir do cache (para o Hugging Face, pode-se definir `HF_HUB_OFFLINE=1`).
6. **Apple Silicon (MPS)** está disponível nesta máquina, mas a execução em MPS ainda
   não foi testada; o teste foi feito em CPU.

## Variantes maiores (não utilizadas)

Base/Large do Depth Anything V2 (relativo e métrico) existem no Hugging Face, porém
com licença **CC-BY-NC-4.0** (uso não comercial) e custo computacional maior. Ficam
fora do escopo da primeira versão.

## Limpeza do cache

```bash
rm -rf models/cache   # os pesos serão baixados novamente na próxima execução
```
