"""
Core diffusion model functionality for text-to-image generation.
Supports multiple Hugging Face diffusion models with optimized performance.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    KandinskyV22Pipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    safety_checker: bool = True
    memory_efficient: bool = False

@dataclass
class ModelInfo:
    """Information about a diffusion model."""
    name: str
    model_id: str
    description: str
    max_resolution: int
    recommended_steps: int
    memory_usage: str  # "low", "medium", "high"

class DiffusionModelManager:
    """Manages diffusion models and image generation."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_pipeline = None
        
        # Available models
        self.models = {
            "stable-diffusion-1.5": ModelInfo(
                name="Stable Diffusion 1.5",
                model_id="runwayml/stable-diffusion-v1-5",
                description="Fast, high-quality generation",
                max_resolution=1024,
                recommended_steps=20,
                memory_usage="medium"
            ),
            "stable-diffusion-2.1": ModelInfo(
                name="Stable Diffusion 2.1",
                model_id="stabilityai/stable-diffusion-2-1",
                description="Enhanced quality and detail",
                max_resolution=1024,
                recommended_steps=25,
                memory_usage="medium"
            ),
            "stable-diffusion-xl": ModelInfo(
                name="Stable Diffusion XL",
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                description="Ultra-high resolution images",
                max_resolution=2048,
                recommended_steps=30,
                memory_usage="high"
            ),
            "kandinsky-2.2": ModelInfo(
                name="Kandinsky 2.2",
                model_id="kandinsky-community/kandinsky-2-2-decoder",
                description="Artistic and creative styles",
                max_resolution=1024,
                recommended_steps=25,
                memory_usage="medium"
            )
        }
        
        logger.info(f"Initialized on device: {self.device}")
    
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get list of available models."""
        return self.models
    
    def load_model(self, model_key: str, memory_efficient: bool = False) -> bool:
        """
        Load a diffusion model.
        
        Args:
            model_key: Key identifying the model to load
            memory_efficient: Whether to use memory-efficient loading
            
        Returns:
            True if successful, False otherwise
        """
        if model_key not in self.models:
            logger.error(f"Unknown model: {model_key}")
            return False
        
        try:
            model_info = self.models[model_key]
            logger.info(f"Loading model: {model_info.name}")
            
            # Clear previous model
            if self.current_pipeline:
                del self.current_pipeline
                torch.cuda.empty_cache()
            
            # Load model based on type
            if "stable-diffusion-xl" in model_key:
                self.current_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                )
            elif "kandinsky" in model_key:
                self.current_pipeline = KandinskyV22Pipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                )
            
            # Move to device
            self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Enable memory efficient attention if requested
            if memory_efficient and hasattr(self.current_pipeline, 'enable_memory_efficient_attention'):
                self.current_pipeline.enable_memory_efficient_attention()
            
            # Enable CPU offloading for memory efficiency
            if memory_efficient and self.device == "cuda":
                self.current_pipeline.enable_sequential_cpu_offload()
            
            # Set scheduler for better quality
            if hasattr(self.current_pipeline, 'scheduler'):
                self.current_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.current_pipeline.scheduler.config
                )
            
            self.current_model = model_key
            logger.info(f"Successfully loaded {model_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {str(e)}")
            return False
    
    def generate_images(self, config: GenerationConfig) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Generate images from text prompt.
        
        Args:
            config: Generation configuration
            
        Returns:
            Tuple of (generated_images, generation_info)
        """
        if not self.current_pipeline:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            # Set seed for reproducibility
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(config.seed)
            
            # Prepare generation arguments
            generation_kwargs = {
                "prompt": config.prompt,
                "negative_prompt": config.negative_prompt,
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "num_images_per_prompt": config.num_images_per_prompt,
            }
            
            # Add safety checker setting if supported
            if hasattr(self.current_pipeline, 'safety_checker'):
                generation_kwargs["safety_checker"] = config.safety_checker
            
            # Generate images
            start_time = time.time()
            result = self.current_pipeline(**generation_kwargs)
            generation_time = time.time() - start_time
            
            # Extract images
            if isinstance(result, dict):
                images = result["images"]
            else:
                images = result
            
            # Prepare generation info
            generation_info = {
                "model": self.current_model,
                "prompt": config.prompt,
                "negative_prompt": config.negative_prompt,
                "width": config.width,
                "height": config.height,
                "steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "seed": config.seed,
                "generation_time": generation_time,
                "device": self.device,
                "num_images": len(images)
            }
            
            logger.info(f"Generated {len(images)} images in {generation_time:.2f}s")
            return images, generation_info
            
        except Exception as e:
            logger.error(f"Image generation dfailed: {str(e)}")
            raise
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the currently loaded model."""
        if self.current_model:
            return self.models[self.current_model]
        return None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "torch_version": torch.__version__
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_cached": torch.cuda.memory_reserved(0)
            })
        
        return info
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def estimate_memory_usage(self, config: GenerationConfig) -> Dict[str, Any]:
        """
        Estimate memory usage for a given configuration.
        
        Args:
            config: Generation configuration
            
        Returns:
            Dictionary with memory usage estimates
        """
        model_info = self.get_model_info()
        if not model_info:
            return {"error": "No model loaded"}
        
        # Base model memory (rough estimates)
        base_memory = {
            "low": 2.5,      # GB
            "medium": 4.0,   # GB
            "high": 8.0      # GB
        }
        
        # Additional memory for generation
        image_pixels = config.width * config.height
        additional_memory = (image_pixels * config.num_images_per_prompt * 4) / (1024**3)  # GB
        
        total_estimated = base_memory.get(model_info.memory_usage, 4.0) + additional_memory
        
        return {
            "base_model_memory": base_memory.get(model_info.memory_usage, 4.0),
            "generation_memory": additional_memory,
            "total_estimated": total_estimated,
            "recommended_gpu_memory": max(6, int(total_estimated * 1.5))
        }
    
    def get_recommended_settings(self, model_key: str) -> Dict[str, Any]:
        """
        Get recommended settings for a model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Dictionary with recommended settings
        """
        if model_key not in self.models:
            return {}
        
        model_info = self.models[model_key]
        
        return {
            "width": min(512, model_info.max_resolution),
            "height": min(512, model_info.max_resolution),
            "steps": model_info.recommended_steps,
            "guidance_scale": 7.5,
            "memory_efficient": model_info.memory_usage in ["medium", "high"]
        }
