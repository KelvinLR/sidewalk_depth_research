"""Configuração central e resolução de caminhos do projeto.

Todos os caminhos são resolvidos a partir da raiz do projeto (o diretório que
contém ``config/settings.yaml``), e não a partir do diretório de execução.
Isso garante o mesmo comportamento no terminal, no VS Code, em notebooks
Jupyter (cujo cwd costuma ser ``notebooks/``) e no pytest.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SETTINGS_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"


class ConfigError(RuntimeError):
    """Erro de configuração (arquivo ausente, chave inexistente etc.)."""


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Carrega o ``settings.yaml``. O resultado é cacheado por caminho."""
    settings_path = Path(path) if path is not None else SETTINGS_PATH
    if not settings_path.is_file():
        raise ConfigError(f"Arquivo de configuração não encontrado: {settings_path}")
    with settings_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuração inválida (esperado um mapeamento): {settings_path}")
    return data


def resolve_path(relative: str | Path) -> Path:
    """Converte um caminho relativo à raiz do projeto em caminho absoluto."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def get_path(key: str, *, create: bool = False) -> Path:
    """Retorna o caminho absoluto de ``paths.<key>`` definido no settings.yaml.

    Args:
        key: chave dentro da seção ``paths`` (ex.: ``"depth_maps_dir"``).
        create: se ``True``, cria o diretório caso não exista.
    """
    paths = load_settings().get("paths", {})
    if key not in paths:
        raise ConfigError(f"Caminho '{key}' não definido em settings.yaml (seção 'paths').")
    resolved = resolve_path(paths[key])
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_model_config(model_key: str) -> dict[str, Any]:
    """Retorna a configuração de um modelo (seção ``models.<model_key>``)."""
    models = load_settings().get("models", {})
    if model_key not in models:
        available = ", ".join(sorted(models)) or "(nenhum)"
        raise ConfigError(f"Modelo '{model_key}' não configurado. Disponíveis: {available}")
    return dict(models[model_key])


def get_huggingface_cache_dir() -> Path:
    """Diretório de cache dos modelos do Hugging Face (``models/cache/huggingface/hub``)."""
    path = get_path("models_cache_dir") / "huggingface" / "hub"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_torch_hub_dir() -> Path:
    """Diretório do ``torch.hub`` (``models/cache/torch/hub``)."""
    path = get_path("models_cache_dir") / "torch" / "hub"
    path.mkdir(parents=True, exist_ok=True)
    return path
