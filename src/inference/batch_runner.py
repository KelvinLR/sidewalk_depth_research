"""Execução de inferência em lote (batch processing) para múltiplas imagens.

Permite processar conjuntos inteiros de imagens (ou subconjuntos selecionados)
reutilizando os modelos já carregados na memória/dispositivo, evitando o overhead
de reinicialização a cada imagem.

Salva para cada imagem:
- Dados brutos (.npz) e metadados (.json);
- Mapas visuais ilustrativos (gray e color);
- Painéis comparativos consolidados (comparison_<imagem>.png);
- Painéis de inspeção de relevo e defeito (defect_inspection e relief_comparison);
- Registro histórico unificado em outputs/reports/inference_log.csv.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, get_path, load_settings
from src.inference.base_estimator import BaseDepthEstimator, DepthResult
from src.inference.model_factory import available_models, create_estimator
from src.inference.results_io import append_inference_log, save_depth_result
from src.preprocessing.image_processor import LoadedImage, load_rgb_image
from src.visualization.depth_visualizer import (
    plot_comparison,
    plot_defect_inspection,
    plot_multi_model_relief_comparison,
    save_depth_visualizations,
    save_figure,
)


@dataclass
class BatchInferenceSummary:
    """Resultado consolidado da execução em lote."""

    results_by_image: dict[str, list[DepthResult]]
    loaded_images: dict[str, LoadedImage]
    summary_df: pd.DataFrame
    saved_artifacts: dict[str, dict[str, Path]]


def run_batch_inference(
    images: Sequence[Path | str],
    models: Sequence[str] | None = None,
    device: str = "auto",
    save_outputs: bool = True,
    save_comparisons: bool = True,
    save_defect_inspection: bool = True,
    defect_model_key: str = "depth_anything_v2_small",
    relief_sigma: float = 25.0,
    warmup_runs: int = 0,
    verbose: bool = True,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> BatchInferenceSummary:
    """Executa a inferência para múltiplas imagens em lote.

    Carrega cada modelo selecionado apenas uma vez e executa as predições
    sequencialmente em todas as imagens fornecidas.

    Args:
        images: Lista de caminhos para as imagens de entrada.
        models: Lista de chaves de modelos (None para todos os disponíveis).
        device: Dispositivo de execução ('auto', 'cpu', 'mps', 'cuda').
        save_outputs: Se True, salva .npz, .json e PNGs de cada resultado.
        save_comparisons: Se True, gera e salva comparison_<imagem>.png.
        save_defect_inspection: Se True, gera painéis de relevo e defeito.
        defect_model_key: Modelo preferencial para o painel de inspeção de defeito.
        relief_sigma: Escala espacial para o filtro passa-alta de relevo.
        warmup_runs: Execuções descartadas para aquecimento do estimador.
        verbose: Se True, imprime mensagens de progresso no stdout.
        progress_callback: Função opcional chamada a cada passo: callback(img_idx, total_imgs, img_name, model_key).

    Returns:
        BatchInferenceSummary contendo dicionário de resultados por imagem, imagens carregadas,
        DataFrame resumo e caminhos de artefatos salvos.
    """
    image_paths = [Path(p).resolve() for p in images]
    if not image_paths:
        raise ValueError("A lista de imagens está vazia.")

    for p in image_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {p}")

    model_keys = list(models) if models is not None else available_models()
    if not model_keys:
        raise ValueError("Nenhum modelo selecionado para inferência.")

    comparisons_dir = get_path("comparisons_dir", create=True)

    # 1. Carrega todos os estimadores uma única vez
    if verbose:
        print(f"--- Carregando {len(model_keys)} modelo(s) no dispositivo '{device}' ---")
    estimators: dict[str, BaseDepthEstimator] = {}
    for mk in model_keys:
        if verbose:
            print(f"Carregando modelo: {mk}...")
        estimators[mk] = create_estimator(mk, device=device).load()

    results_by_image: dict[str, list[DepthResult]] = {}
    loaded_images: dict[str, LoadedImage] = {}
    summary_rows: list[dict[str, Any]] = []
    saved_artifacts: dict[str, dict[str, Path]] = {}

    total_images = len(image_paths)

    # 2. Loop de processamento de cada imagem
    for img_idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem
        if verbose:
            print(f"\n[{img_idx}/{total_images}] Processando: {img_path.name} ({img_path.parent.name})")

        loaded_img = load_rgb_image(img_path)
        loaded_images[stem] = loaded_img
        img_results: list[DepthResult] = []
        image_artifacts: dict[str, Path] = {}

        # Executa cada modelo na imagem atual
        for mk, estimator in estimators.items():
            if progress_callback:
                progress_callback(img_idx, total_images, img_path.name, mk)

            res = estimator.predict(loaded_img.rgb, warmup_runs=warmup_runs)
            img_results.append(res)

            stats = res.stats()
            summary_rows.append(
                {
                    "Imagem": img_path.name,
                    "Categoria": loaded_img.category or img_path.parent.name,
                    "Dimensões": f"{loaded_img.rgb.shape[1]}x{loaded_img.rgb.shape[0]}",
                    "Modelo": res.model_name,
                    "Chave": res.model_key,
                    "Natureza da Saída": res.output_type.value,
                    "Unidade": res.units or "adimensional",
                    "Tempo Inferência (ms)": round(res.inference_time_s * 1000, 1),
                    "Tempo Total (ms)": round(res.total_time_s * 1000, 1),
                    "Tensor Entrada": "x".join(map(str, res.model_input_shape)),
                    "Resolução Saída": f"{res.output_shape[1]}x{res.output_shape[0]}",
                    "Mín": round(stats["min"], 3),
                    "Máx": round(stats["max"], 3),
                    "Mediana": round(stats["median"], 3),
                    "Device": res.device,
                }
            )

            # Persistência individual do par (imagem, modelo)
            if save_outputs:
                npz_path = save_depth_result(res, img_path)
                vis_paths = save_depth_visualizations(res, npz_path.parent)
                append_inference_log(res, img_path, artifact=npz_path)
                image_artifacts[f"{mk}_npz"] = npz_path
                image_artifacts[f"{mk}_color"] = vis_paths["color"]

            if verbose:
                print(f"  ✓ {res.model_name:38s} | {res.inference_time_s * 1000:6.1f} ms | Saída: {stats['min']:.2f} a {stats['max']:.2f} {res.units or ''}")

        results_by_image[stem] = img_results

        # 3. Painéis comparativos por imagem
        if save_comparisons and img_results:
            comp_fig = plot_comparison(
                loaded_img.rgb,
                img_results,
                title=f"Sidewalk Depth — Comparação de Modelos ({img_path.name})",
            )
            comp_path = comparisons_dir / f"comparison_{stem}.png"
            save_figure(comp_fig, comp_path)
            image_artifacts["comparison_figure"] = comp_path
            matplotlib.pyplot.close(comp_fig)
            if verbose:
                print(f"  ✓ Painel comparativo salvo: {comp_path.name}")

        # 4. Inspeção de defeito e relevo residual por imagem
        if save_defect_inspection and img_results:
            # Seleciona o modelo para a inspeção de 4 vistas
            target_res = next((r for r in img_results if r.model_key == defect_model_key), img_results[0])
            defect_fig = plot_defect_inspection(
                loaded_img.rgb,
                target_res,
                sigma=relief_sigma,
                title=f"Inspeção de Defeito — {target_res.model_name} ({img_path.name})",
            )
            defect_path = comparisons_dir / f"defect_inspection_{stem}_{target_res.model_key}.png"
            save_figure(defect_fig, defect_path)
            image_artifacts["defect_inspection_figure"] = defect_path
            matplotlib.pyplot.close(defect_fig)

            relief_comp_fig = plot_multi_model_relief_comparison(
                loaded_img.rgb,
                img_results,
                sigma=relief_sigma,
                title=f"Comparação de Relevo Residual Local (σ={relief_sigma:.0f}px) — {img_path.name}",
            )
            relief_comp_path = comparisons_dir / f"relief_comparison_{stem}.png"
            save_figure(relief_comp_fig, relief_comp_path)
            image_artifacts["relief_comparison_figure"] = relief_comp_path
            matplotlib.pyplot.close(relief_comp_fig)
            if verbose:
                print(f"  ✓ Painéis de relevo e defeito salvos: {defect_path.name}, {relief_comp_path.name}")

        saved_artifacts[stem] = image_artifacts

    summary_df = pd.DataFrame(summary_rows)

    if verbose:
        print(f"\n✓ Processamento em lote concluído com sucesso: {total_images} imagem(ns), {len(summary_rows)} inferência(s) total.")

    return BatchInferenceSummary(
        results_by_image=results_by_image,
        loaded_images=loaded_images,
        summary_df=summary_df,
        saved_artifacts=saved_artifacts,
    )


def main() -> None:
    """CLI para execução em lote via linha de comando."""
    from src.preprocessing.image_processor import list_dataset_images

    parser = argparse.ArgumentParser(description="Execução em lote de modelos de estimativa de profundidade.")
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default="uneven_surfaces",
        help="Categoria do dataset a processar (padrão: uneven_surfaces)",
    )
    parser.add_argument(
        "--images",
        "-i",
        nargs="+",
        type=str,
        default=None,
        help="Caminhos de imagens específicas (se omitido, processa todas da categoria)",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="Modelos a executar (se omitido, executa todos os habilitados)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Dispositivo de execução ('auto', 'cpu', 'mps', 'cuda')",
    )

    args = parser.parse_args()

    if args.images:
        target_images = [Path(p) for p in args.images]
    else:
        target_images = list_dataset_images(args.category)

    print(f"Iniciando processamento em lote de {len(target_images)} imagem(ns)...")
    summary = run_batch_inference(
        images=target_images,
        models=args.models,
        device=args.device,
        verbose=True,
    )
    print("\nTabela Resumo:")
    print(summary.summary_df[["Imagem", "Modelo", "Tempo Inferência (ms)", "Mín", "Máx", "Device"]].to_string(index=False))


if __name__ == "__main__":
    main()
