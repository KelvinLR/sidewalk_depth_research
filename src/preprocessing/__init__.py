"""Leitura de imagens RGB e pré-processamento independente de modelo."""

from src.preprocessing.image_processor import ImageLoadError, LoadedImage, list_dataset_images, load_rgb_image

__all__ = ["ImageLoadError", "LoadedImage", "list_dataset_images", "load_rgb_image"]
