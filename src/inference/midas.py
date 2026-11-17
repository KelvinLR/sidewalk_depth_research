"""MiDaS v2.1 Small via ``torch.hub``.

Diferenças em relação ao Depth Anything (por isso o pré-processamento NÃO é
compartilhado):

- Carregado do repositório oficial via ``torch.hub`` (tag fixa ``v3_1``), não
  pelo Transformers.
- Pré-processamento próprio (``transforms.small_transform`` do repositório):
  entrada RGB ``(H, W, 3)`` -> ``/255`` -> lado maior para 256 px mantendo a
  proporção, dimensões múltiplas de 32, bicúbica (OpenCV) -> normalização
  ImageNet -> tensor ``(1, 3, h, w)``.
- Backbone EfficientNet-Lite3, cujo construtor chama
  ``torch.hub.load("rwightman/gen-efficientnet-pytorch", ...)`` **sem**
  ``trust_repo``. Em ambiente novo isso abre um prompt interativo ``(y/N)`` que
  trava notebooks; por isso esse repositório é pré-autorizado antes do carregamento.
- Saída: disparidade relativa afim-invariante (maior = mais perto), com escala
  numérica arbitrária (tipicamente centenas). Não é distância.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from src.config import get_torch_hub_dir
from src.inference.base_estimator import BaseDepthEstimator, DepthOutputType, EstimatorError


class MiDaSEstimator(BaseDepthEstimator):
    """Estimador para modelos MiDaS carregados via ``torch.hub``."""

    def __init__(self, model_key: str, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(model_key, config, **kwargs)
        missing = [k for k in ("hub_repo", "hub_model", "hub_transform") if k not in self.config]
        if missing:
            raise EstimatorError(f"{model_key}: chaves ausentes na configuração: {missing}")
        if self.output_type is not DepthOutputType.RELATIVE_INVERSE_DEPTH:
            raise EstimatorError(f"{model_key}: MiDaS produz apenas disparidade relativa.")
        self.hub_repo: str = self.config["hub_repo"]
        self.hub_model: str = self.config["hub_model"]
        self.hub_transform: str = self.config["hub_transform"]
        self.hub_dependencies: list[str] = list(self.config.get("hub_dependencies", []))
        self.model: torch.nn.Module | None = None
        self.transform: Any = None

    def _load(self) -> None:
        torch.hub.set_dir(str(get_torch_hub_dir()))

        # Monkey-patch para evitar que torch.hub faça requisições HTTP ao GitHub
        # apenas para descobrir o branch padrão (main vs master), o que causa HTTP 403
        # quando rodando offline, em ambiente restrito ou sob rate-limiting da API do GitHub.
        orig_parse_repo_info = torch.hub._parse_repo_info

        def _cache_aware_parse_repo_info(github: str):
            if ":" not in github:
                parts = github.split("/")
                if len(parts) == 2:
                    owner, name = parts
                    hub_dir = torch.hub.get_dir()
                    for ref in ("master", "main"):
                        if (get_torch_hub_dir() / f"{owner}_{name}_{ref}").is_dir():
                            return owner, name, ref
            return orig_parse_repo_info(github)

        torch.hub._parse_repo_info = _cache_aware_parse_repo_info
        try:
            with warnings.catch_warnings():
                # timm e torch.jit emitem warnings ao serem importados pelo hubconf do MiDaS.
                warnings.filterwarnings("ignore", category=FutureWarning)
                for dep in self.hub_dependencies:
                    self._trust_hub_repo(dep)
                model = torch.hub.load(self.hub_repo, self.hub_model, trust_repo=True)
                transforms = torch.hub.load(self.hub_repo, "transforms", trust_repo=True)
        finally:
            torch.hub._parse_repo_info = orig_parse_repo_info

        if not hasattr(transforms, self.hub_transform):
            raise EstimatorError(f"{self.model_key}: transformação '{self.hub_transform}' não encontrada no hub.")
        self.transform = getattr(transforms, self.hub_transform)
        self.model = model.to(self.device).eval()

    @staticmethod
    def _trust_hub_repo(repo: str) -> None:
        """Pré-autoriza um repositório do torch.hub (evita o prompt interativo).

        ``torch.hub.list(..., trust_repo=True)`` é a API pública que registra o
        repositório em ``<hub_dir>/trusted_list`` (e baixa o código, se necessário).
        Sem ref explícita usa ``master``, que é o branch usado pelo MiDaS.
        """
        ref = repo if ":" in repo else f"{repo}:master"
        torch.hub.list(ref, trust_repo=True, verbose=False)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        # small_transform espera RGB (H, W, 3) e divide por 255 internamente.
        return self.transform(image).to(self.device)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(inputs)  # (1, h, w)

    def _input_shape(self, inputs: torch.Tensor) -> tuple[int, ...]:
        return tuple(inputs.shape)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["preprocessing"] = {
            "transform": f"{self.hub_repo}:transforms.{self.hub_transform}",
            "input_scaling": "rgb / 255",
            "resize": "lado maior -> 256, mantém proporção, múltiplo de 32, bicúbica (OpenCV)",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        return info
