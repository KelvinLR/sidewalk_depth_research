# Sidewalk Depth Research

Ambiente experimental em Python para investigar a viabilidade de **identificar e
mensurar irregularidades em calçadas** (buracos, desníveis entre placas, elevações,
rampas, obstáculos) com **estimativa monocular de profundidade** e visão
computacional clássica.

> [!NOTE]
> Este é um ambiente de pesquisa, não um produto. Ele é independente do aplicativo
> Android; os resultados servirão para orientar uma futura implementação embarcada
> (Kotlin + Jetpack Compose). Não há treinamento ou fine-tuning: apenas modelos
> pré-treinados.

## Status

| Parte | Conteúdo | Status |
|---|---|---|
| 1–4 | Contexto, objetivos, stack, estrutura, ambiente, configuração | ✅ Concluída |
| — | Investigação de compatibilidade dos 3 modelos | ✅ Testados em CPU e Apple Silicon (MPS) — ver [models/README.md](models/README.md) |
| 5 | Implementação dos estimadores (`src/inference/`) | ✅ Concluída (`DepthAnythingV2`, `MiDaS`, `model_factory`, `results_io`) |
| 6 | Notebook `01_model_inference.ipynb` | ✅ Concluído e executado de ponta a ponta |
| 7 | Pré-processamento e interpretação dos resultados | ✅ Concluído (`load_rgb_image`, orientação EXIF, semântica relativa vs métrica) |
| 8–9 | Preparação geométrica, dataset e medições | 🟡 Estrutura, CSV e documentação de requisitos prontos |
| — | Validação com fotografia real de calçada | ✅ Concluída (`broken_sidewalk_la_01.jpg` em `uneven_surfaces/`) |

## Modelos

| Chave (`settings.yaml`) | Modelo | Fonte | Saída |
|---|---|---|---|
| `depth_anything_v2_small` | Depth Anything V2 Small | Hugging Face Transformers | Disparidade **relativa** (sem unidade; maior = mais perto) |
| `depth_anything_v2_metric_outdoor_small` | Depth Anything V2 Metric Outdoor Small | Hugging Face Transformers | Profundidade **métrica** (metros; maior = mais longe) |
| `midas_v21_small` | MiDaS v2.1 Small | `torch.hub` (`intel-isl/MiDaS:v3_1`) | Disparidade **relativa** (sem unidade; maior = mais perto) |
| `yolo26n_depth` | YOLO26n Depth | Ultralytics (`yolo26n-depth.pt`) | Profundidade **métrica** (metros; maior = mais longe) |

> [!WARNING]
> **Profundidade relativa não é distância.** Saídas relativas são afim-invariantes
> (escala e deslocamento desconhecidos) e representam o *inverso* da profundidade.
> Nunca as interprete como metros ou centímetros. Mesmo a saída métrica precisa ser
> validada: o modelo foi ajustado em cenas de rua sintéticas (Virtual KITTI 2, até
> 80 m), não em calçadas fotografadas a 1–3 m.

Detalhes de pré-processamento, formatos de tensor e armadilhas conhecidas:
[models/README.md](models/README.md).

## Arquitetura do projeto

```text
sidewalk_depth_research/
├── config/settings.yaml        # configuração central (caminhos, device, modelos)
├── datasets/
│   ├── images/<categoria>/     # regular, potholes, uneven_surfaces, ramps, challenging_conditions
│   └── measurements.csv        # medições físicas de referência (apenas cabeçalho)
├── models/                     # README + cache local de pesos (models/cache/, não versionado)
├── notebooks/                  # 01..04 (01 é o entregável principal da v1)
├── outputs/                    # depth_maps/, comparisons/, reports/ (não versionados)
├── src/
│   ├── config.py               # carrega settings.yaml e resolve caminhos a partir da raiz
│   ├── inference/              # estimadores (base, Depth Anything, MiDaS, factory)
│   ├── preprocessing/          # leitura de imagens, orientação EXIF, RGB
│   ├── visualization/          # visualização (separada da inferência)
│   ├── geometry/               # reservado: nuvem de pontos, RANSAC, planos
│   └── evaluation/             # reservado: validação contra medições
├── tests/                      # pytest
├── .vscode/settings.json       # interpretador .venv, cwd dos notebooks = raiz
├── pyproject.toml              # apenas configuração de ferramentas (pytest)
└── requirements.txt            # dependências com versões fixadas
```

### Decisões de estrutura

- **`src/` como pacote importável a partir da raiz** (`from src.config import ...`),
  sem instalação: o pytest usa `pythonpath = ["."]` e o VS Code executa os notebooks
  com `jupyter.notebookFileRoot = ${workspaceFolder}`.
- **Caminhos independentes do diretório de execução**: todos os caminhos do
  `settings.yaml` são relativos à raiz e resolvidos por
  [`src/config.py`](src/config.py) a partir da localização do próprio arquivo.
- **`src/config.py` e `pyproject.toml`** foram adicionados à estrutura proposta;
  `.vscode/` foi incluído para garantir o comportamento no VS Code. Nenhuma outra
  camada foi criada.
- **Cache de pesos dentro do projeto** (`models/cache/`), para não baixar novamente a
  cada execução e manter o experimento autocontido.

## Requisitos de ambiente

- Python **3.11** (versão testada).
- ~2 GB de disco para o ambiente virtual + ~330 MB para os pesos.
- Internet apenas na **primeira** execução (download dos pesos).
- GPU **opcional**. Funciona em CPU; CUDA (NVIDIA) e MPS (Apple Silicon) são
  detectados automaticamente (`runtime.device: auto`).

## Criação do ambiente virtual e instalação

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No VS Code: abra a pasta do projeto, selecione o interpretador `.venv` e o kernel
`.venv` nos notebooks.

### GPU NVIDIA com CUDA (opcional)

O `torch` padrão do PyPI em Linux já inclui suporte a CUDA para uma versão específica;
no Windows, o wheel padrão é **somente CPU**. Para usar GPU, instale o PyTorch com
CUDA **antes** do `requirements.txt`, usando o seletor oficial em
<https://pytorch.org/get-started/locally/>. Exemplo (ajuste `cuXXX` à versão
suportada pelo seu driver):

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cuXXX
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

> [!NOTE]
> A configuração com CUDA não foi testada neste projeto (desenvolvido em macOS M1).

### Decisões de dependências

Versões fixadas no `requirements.txt` correspondem ao conjunto efetivamente
instalado e testado em 2026-11-16.

| Pacote | Por quê |
|---|---|
| `torch`, `transformers` | Execução dos modelos; Depth Anything V2 é suportado nativamente pelo Transformers. |
| `torchvision`, `timm` | **Exigidos pelo `hubconf.py` do MiDaS**, mesmo para o modelo Small. O `timm` está fixado porque versões futuras podem quebrar o MiDaS (já há `FutureWarning`). |
| `opencv-python-headless` | Processamento de imagens sem GUI/Qt (visualização fica com Matplotlib; evita conflitos em servidores e notebooks). |
| `numpy`, `pillow`, `matplotlib`, `pandas` | Operações matriciais, leitura de imagem/EXIF, visualização, tabelas de resultados. |
| `open3d` | Reservado para análise geométrica futura. Instala dependências pesadas (Dash, Plotly, Flask); o plugin pytest do Dash é desativado em `pyproject.toml`. |
| `pyyaml` | Leitura de `config/settings.yaml`. |
| `ipykernel` | Kernel para notebooks no VS Code. Para usar JupyterLab fora do VS Code: `pip install jupyterlab`. |
| `pytest` | Testes. |

## Obtenção dos pesos

Automática na primeira execução, com cache em `models/cache/`. Ver
[models/README.md](models/README.md).

## Organização do dataset

Coloque as imagens em `datasets/images/<categoria>/`:

| Pasta | Conteúdo |
|---|---|
| `regular/` | Calçadas sem irregularidades relevantes (referência) |
| `potholes/` | Buracos e depressões |
| `uneven_surfaces/` | Desníveis entre placas, elevações, raízes |
| `ramps/` | Rampas e inclinações |
| `challenging_conditions/` | Sombra forte, chuva/superfície molhada, baixa luz, reflexos |

As imagens não são versionadas por padrão (ver `.gitignore`), pois podem ser grandes
ou ter restrições de licença/privacidade. Ao usar datasets públicos, registre a
origem e a licença em `notes`.

### `datasets/measurements.csv`

Uma linha por imagem com referência física. **Campos de medição vazios = ausentes**
(lidos como `NaN`, nunca como zero). Não preencha valores estimados.

| Coluna | Tipo / unidade | Obrigatória |
|---|---|---|
| `image_name` | nome do arquivo (ex.: `IMG_0001.jpg`) | sim |
| `category` | uma das pastas acima | sim |
| `location_id` | identificador do local | não |
| `real_depth_cm` | profundidade do buraco/depressão (cm) | não |
| `real_height_difference_cm` | desnível/elevação (cm) | não |
| `real_width_cm`, `real_length_cm` | dimensões da irregularidade (cm) | não |
| `camera_height_cm` | altura da câmera ao chão (cm) | não |
| `camera_distance_cm` | distância horizontal câmera → irregularidade (cm) | não |
| `camera_angle` | inclinação da câmera (graus; documentar a convenção em `notes`) | não |
| `lighting_condition` | ex.: `sol`, `sombra`, `nublado`, `noite` | não |
| `pavement_type` | ex.: `concreto`, `ladrilho`, `pedra portuguesa`, `asfalto` | não |
| `notes` | observações livres, origem/licença da imagem | não |

## Execução dos notebooks

### Pelo VS Code
1. Abra a pasta do projeto no VS Code: `code .`
2. Abra `notebooks/01_model_inference.ipynb`.
3. No canto superior direito do editor de notebooks, selecione o kernel do ambiente virtual: **`.venv (Python 3.11)`**.
4. Clique em **"Run All"** (Executar Tudo).
5. Para testar com outra imagem, basta colocar a foto em qualquer pasta de `datasets/images/<categoria>/` e alterar `IMAGE_INDEX` ou `SELECTED_IMAGE_PATH` na Célula 2.

### Pelo terminal (JupyterLab ou script)
```bash
# Ativar o ambiente virtual
source .venv/bin/activate

# Executar os testes automatizados
pytest

# Opcional: abrir via JupyterLab
pip install jupyterlab
jupyter lab notebooks/01_model_inference.ipynb
```

## Localização dos resultados

| Pasta | Conteúdo |
|---|---|
| `outputs/depth_maps/` | Mapas de profundidade (saída bruta preservada + versões para visualização) |
| `outputs/comparisons/` | Figuras lado a lado entre modelos |
| `outputs/reports/` | Tabelas e relatórios (tempo de inferência, metadados) |

## Testes

```bash
pytest            # testes básicos (estrutura, configuração, esquema do CSV)
```

## Limitações conhecidas

- Mapas relativos (Depth Anything V2 Small, MiDaS) **não permitem medir** dimensões
  físicas sem recuperação de escala/deslocamento e calibração.
- O modelo Metric Outdoor está fora do seu domínio de treino em fotos de calçada a
  curta distância; seus valores em metros ainda não foram validados.
- O MiDaS depende de código baixado via `torch.hub` (incluindo um repositório não
  fixável em versão) e exige pré-autorização para não travar em prompt interativo.
- Até o momento, os modelos só foram executados com imagem **sintética** em CPU;
  não houve validação com fotografias reais, MPS ou CUDA.
