"""
AI Image Generator

A powerful text-to-image generation application using Hugging Face's diffusion models.
Create stunning images from text descriptions with state-of-the-art AI models.
"""

from .diffusion_models import DiffusionModelManager, GenerationConfig, ModelInfo
from .image_processor import ImageProcessor

__version__ = "1.0.0"
__author__ = "AI Image Generator"
__email__ = ""

__all__ = [
    "DiffusionModelManager",
    "GenerationConfig", 
    "ModelInfo",
    "ImageProcessor"
]
