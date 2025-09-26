#!/usr/bin/env python3
"""
Model Evaluator for Diffusion Models

A comprehensive evaluation script that downloads models from Hugging Face
and runs various evaluations from basic to advanced.

Usage:
    python model_evaluator.py --model "runwayml/stable-diffusion-v1-5" --eval-type all
    python model_evaluator.py --model "stabilityai/stable-diffusion-2-1" --eval-type basic
    python model_evaluator.py --model "stabilityai/stable-diffusion-xl-base-1.0" --eval-type advanced
"""

import os
import sys
import time
import json
import argparse
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import psutil
from tqdm import tqdm

# Hugging Face imports
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)
from transformers import CLIPProcessor, CLIPModel
import accelerate

# Custom imports
from diffusion_models import DiffusionModelManager, GenerationConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    model_name: str
    model_size_mb: float
    inference_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    image_quality_score: float
    prompt_adherence_score: float
    style_consistency_score: float
    diversity_score: float
    aesthetic_score: float
    technical_score: float
    overall_score: float
    timestamp: str

@dataclass
class TestPrompt:
    """Test prompt with expected characteristics"""
    prompt: str
    category: str
    difficulty: str  # easy, medium, hard
    expected_style: str
    key_elements: List[str]

class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize CLIP for semantic evaluation
        self.clip_model = None
        self.clip_processor = None
        self._setup_clip()
        
        # Test prompts for evaluation
        self.test_prompts = self._create_test_prompts()
        
        # Evaluation results
        self.results: List[EvaluationMetrics] = []
        
    def _setup_clip(self):
        """Initialize CLIP model for semantic evaluation"""
        try:
            logger.info("Loading CLIP model for semantic evaluation...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _create_test_prompts(self) -> List[TestPrompt]:
        """Create comprehensive test prompts for evaluation"""
        return [
            # Basic prompts
            TestPrompt(
                prompt="a red apple on a white table",
                category="basic",
                difficulty="easy",
                expected_style="photorealistic",
                key_elements=["apple", "red", "white table"]
            ),
            TestPrompt(
                prompt="a beautiful sunset over mountains",
                category="basic",
                difficulty="easy",
                expected_style="landscape",
                key_elements=["sunset", "mountains"]
            ),
            TestPrompt(
                prompt="a cute golden retriever puppy playing in a garden",
                category="basic",
                difficulty="easy",
                expected_style="photorealistic",
                key_elements=["puppy", "golden retriever", "garden"]
            ),
            
            # Medium complexity
            TestPrompt(
                prompt="a futuristic city skyline at night with flying cars and neon lights",
                category="complex",
                difficulty="medium",
                expected_style="futuristic",
                key_elements=["city", "skyline", "flying cars", "neon lights"]
            ),
            TestPrompt(
                prompt="a medieval knight in shining armor riding a dragon",
                category="fantasy",
                difficulty="medium",
                expected_style="fantasy art",
                key_elements=["knight", "armor", "dragon"]
            ),
            TestPrompt(
                prompt="a vintage 1950s diner with chrome details and checkered floors",
                category="complex",
                difficulty="medium",
                expected_style="vintage",
                key_elements=["diner", "1950s", "chrome", "checkered"]
            ),
            
            # Advanced prompts
            TestPrompt(
                prompt="a surreal painting in the style of Salvador Dali featuring melting clocks and floating elephants",
                category="artistic",
                difficulty="hard",
                expected_style="surreal",
                key_elements=["melting clocks", "elephants", "Dali style"]
            ),
            TestPrompt(
                prompt="a hyper-detailed steampunk laboratory with brass gears, copper pipes, and Victorian machinery",
                category="complex",
                difficulty="hard",
                expected_style="steampunk",
                key_elements=["steampunk", "brass gears", "copper pipes", "Victorian"]
            ),
            TestPrompt(
                prompt="a photorealistic portrait of an elderly woman with deep wrinkles and kind eyes, oil painting style",
                category="portrait",
                difficulty="hard",
                expected_style="realistic",
                key_elements=["portrait", "elderly woman", "wrinkles", "oil painting"]
            )
        ]
    
    def download_model(self, model_id: str, cache_dir: Optional[str] = None) -> str:
        """Download model from Hugging Face"""
        logger.info(f"📥 Downloading model: {model_id}")
        
        try:
            # Determine pipeline type based on model
            if "xl" in model_id.lower() or "sd-xl" in model_id.lower():
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline
            
            # Download and cache model
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=cache_dir,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Get model size
            model_size = self._get_model_size(pipeline)
            logger.info(f"✅ Model downloaded successfully ({model_size:.1f} MB)")
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise
    
    def _get_model_size(self, pipeline) -> float:
        """Calculate model size in MB"""
        total_size = 0
        for component in [pipeline.unet, pipeline.vae, pipeline.text_encoder]:
            if hasattr(component, 'parameters'):
                for param in component.parameters():
                    total_size += param.numel() * param.element_size()
        return total_size / (1024 * 1024)  # Convert to MB
    
    def run_basic_evaluation(self, model_id: str, num_samples: int = 5) -> Dict[str, Any]:
        """Run basic performance and quality evaluations"""
        logger.info("🔍 Running basic evaluation...")
        
        metrics = {
            "inference_times": [],
            "memory_usage": [],
            "image_qualities": [],
            "success_rate": 0
        }
        
        try:
            # Load model
            if "xl" in model_id.lower():
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Test with simple prompts
            test_prompts = [tp.prompt for tp in self.test_prompts[:3]]  # Use first 3 basic prompts
            
            for i, prompt in enumerate(tqdm(test_prompts, desc="Basic evaluation")):
                try:
                    # Monitor memory
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Generate image
                    start_time = time.time()
                    result = pipeline(
                        prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        width=512,
                        height=512
                    )
                    end_time = time.time()
                    
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Record metrics
                    inference_time = end_time - start_time
                    metrics["inference_times"].append(inference_time)
                    metrics["memory_usage"].append(memory_after - memory_before)
                    
                    # Basic image quality check
                    if result.images and len(result.images) > 0:
                        image_quality = self._calculate_basic_image_quality(result.images[0])
                        metrics["image_qualities"].append(image_quality)
                        metrics["success_rate"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to generate image for prompt {i}: {e}")
            
            metrics["success_rate"] = metrics["success_rate"] / len(test_prompts)
            
        except Exception as e:
            logger.error(f"Basic evaluation failed: {e}")
            traceback.print_exc()
        
        return metrics
    
    def run_advanced_evaluation(self, model_id: str, num_samples: int = 9) -> Dict[str, Any]:
        """Run advanced semantic and style evaluations"""
        logger.info("🎯 Running advanced evaluation...")
        
        metrics = {
            "prompt_adherence": [],
            "style_consistency": [],
            "diversity_scores": [],
            "aesthetic_scores": [],
            "technical_scores": []
        }
        
        try:
            # Load model
            if "xl" in model_id.lower():
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Test with all prompts
            for test_prompt in tqdm(self.test_prompts, desc="Advanced evaluation"):
                try:
                    # Generate multiple variations
                    images = []
                    for seed in range(3):  # 3 variations per prompt
                        result = pipeline(
                            test_prompt.prompt,
                            num_inference_steps=30,
                            guidance_scale=8.0,
                            width=512,
                            height=512,
                            generator=torch.Generator().manual_seed(seed)
                        )
                        if result.images:
                            images.extend(result.images)
                    
                    if images:
                        # Evaluate prompt adherence
                        adherence = self._evaluate_prompt_adherence(images[0], test_prompt)
                        metrics["prompt_adherence"].append(adherence)
                        
                        # Evaluate style consistency
                        consistency = self._evaluate_style_consistency(images, test_prompt)
                        metrics["style_consistency"].append(consistency)
                        
                        # Evaluate diversity
                        diversity = self._evaluate_diversity(images)
                        metrics["diversity_scores"].append(diversity)
                        
                        # Evaluate aesthetic quality
                        aesthetic = self._evaluate_aesthetic_quality(images[0])
                        metrics["aesthetic_scores"].append(aesthetic)
                        
                        # Evaluate technical quality
                        technical = self._evaluate_technical_quality(images[0])
                        metrics["technical_scores"].append(technical)
                
                except Exception as e:
                    logger.warning(f"Advanced evaluation failed for prompt: {test_prompt.prompt[:50]}... - {e}")
        
        except Exception as e:
            logger.error(f"Advanced evaluation failed: {e}")
            traceback.print_exc()
        
        return metrics
    
    def _calculate_basic_image_quality(self, image: Image.Image) -> float:
        """Calculate basic image quality metrics"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate sharpness (Laplacian variance)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            laplacian_var = np.var(np.array(Image.fromarray(gray).filter(ImageFilter.Laplacian)))
            
            # Calculate brightness and contrast
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Normalize metrics (simple scoring)
            sharpness_score = min(laplacian_var / 1000, 1.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(contrast / 100, 1.0)
            
            return (sharpness_score + brightness_score + contrast_score) / 3.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate image quality: {e}")
            return 0.5
    
    def _evaluate_prompt_adherence(self, image: Image.Image, test_prompt: TestPrompt) -> float:
        """Evaluate how well the image matches the prompt using CLIP"""
        if not self.clip_model or not self.clip_processor:
            return 0.5  # Default score if CLIP not available
        
        try:
            # Encode image and text
            inputs = self.clip_processor(
                text=[test_prompt.prompt], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = torch.nn.functional.cosine_similarity(
                    outputs.image_embeds, outputs.text_embeds
                ).item()
            
            return max(0, similarity)  # Ensure positive score
            
        except Exception as e:
            logger.warning(f"Prompt adherence evaluation failed: {e}")
            return 0.5
    
    def _evaluate_style_consistency(self, images: List[Image.Image], test_prompt: TestPrompt) -> float:
        """Evaluate consistency across multiple generated images"""
        if len(images) < 2:
            return 1.0
        
        try:
            # Calculate feature similarity between images
            similarities = []
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if self.clip_model and self.clip_processor:
                        inputs = self.clip_processor(
                            images=[images[i], images[j]], 
                            return_tensors="pt", 
                            padding=True
                        )
                        with torch.no_grad():
                            outputs = self.clip_model(**inputs)
                            similarity = torch.nn.functional.cosine_similarity(
                                outputs.image_embeds[0:1], outputs.image_embeds[1:2]
                            ).item()
                            similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.warning(f"Style consistency evaluation failed: {e}")
            return 0.5
    
    def _evaluate_diversity(self, images: List[Image.Image]) -> float:
        """Evaluate diversity across generated images"""
        if len(images) < 2:
            return 0.0
        
        try:
            # Calculate feature diversity
            features = []
            for image in images:
                if self.clip_model and self.clip_processor:
                    inputs = self.clip_processor(images=[image], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        outputs = self.clip_model(**inputs)
                        features.append(outputs.image_embeds.numpy())
            
            if features:
                features = np.vstack(features)
                # Calculate pairwise distances
                distances = []
                for i in range(len(features)):
                    for j in range(i + 1, len(features)):
                        dist = np.linalg.norm(features[i] - features[j])
                        distances.append(dist)
                
                return np.mean(distances) if distances else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Diversity evaluation failed: {e}")
            return 0.0
    
    def _evaluate_aesthetic_quality(self, image: Image.Image) -> float:
        """Evaluate aesthetic quality of the image"""
        try:
            # Simple aesthetic scoring based on composition and colors
            img_array = np.array(image)
            
            # Color diversity
            colors = img_array.reshape(-1, img_array.shape[-1])
            unique_colors = len(np.unique(colors.view(np.void, colors.dtype)))
            color_score = min(unique_colors / 10000, 1.0)
            
            # Edge density (composition)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            edges = np.array(Image.fromarray(gray).filter(ImageFilter.FIND_EDGES))
            edge_density = np.mean(edges > 50)
            composition_score = min(edge_density * 2, 1.0)
            
            return (color_score + composition_score) / 2.0
            
        except Exception as e:
            logger.warning(f"Aesthetic evaluation failed: {e}")
            return 0.5
    
    def _evaluate_technical_quality(self, image: Image.Image) -> float:
        """Evaluate technical quality (sharpness, noise, artifacts)"""
        return self._calculate_basic_image_quality(image)
    
    def generate_report(self, model_id: str, basic_metrics: Dict, advanced_metrics: Dict) -> EvaluationMetrics:
        """Generate comprehensive evaluation report"""
        logger.info("📊 Generating evaluation report...")
        
        # Calculate overall metrics
        avg_inference_time = np.mean(basic_metrics.get("inference_times", [0]))
        avg_memory_usage = np.mean(basic_metrics.get("memory_usage", [0]))
        avg_image_quality = np.mean(basic_metrics.get("image_qualities", [0]))
        
        avg_prompt_adherence = np.mean(advanced_metrics.get("prompt_adherence", [0]))
        avg_style_consistency = np.mean(advanced_metrics.get("style_consistency", [0]))
        avg_diversity = np.mean(advanced_metrics.get("diversity_scores", [0]))
        avg_aesthetic = np.mean(advanced_metrics.get("aesthetic_scores", [0]))
        avg_technical = np.mean(advanced_metrics.get("technical_scores", [0]))
        
        # Calculate overall score (weighted average)
        overall_score = (
            avg_image_quality * 0.2 +
            avg_prompt_adherence * 0.25 +
            avg_style_consistency * 0.15 +
            avg_diversity * 0.1 +
            avg_aesthetic * 0.15 +
            avg_technical * 0.15
        )
        
        # Create metrics object
        metrics = EvaluationMetrics(
            model_name=model_id,
            model_size_mb=0.0,  # Will be filled by download_model
            inference_time=avg_inference_time,
            memory_peak_mb=avg_memory_usage,
            memory_avg_mb=avg_memory_usage,
            image_quality_score=avg_image_quality,
            prompt_adherence_score=avg_prompt_adherence,
            style_consistency_score=avg_style_consistency,
            diversity_score=avg_diversity,
            aesthetic_score=avg_aesthetic,
            technical_score=avg_technical,
            overall_score=overall_score,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save detailed results
        self._save_detailed_results(model_id, basic_metrics, advanced_metrics, metrics)
        
        return metrics
    
    def _save_detailed_results(self, model_id: str, basic_metrics: Dict, advanced_metrics: Dict, summary: EvaluationMetrics):
        """Save detailed results to files"""
        # Clean model name for filename
        clean_name = model_id.replace("/", "_").replace("-", "_")
        
        # Save summary
        summary_file = self.output_dir / f"{clean_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        # Save detailed metrics
        detailed_file = self.output_dir / f"{clean_name}_detailed.json"
        detailed_data = {
            "model_id": model_id,
            "basic_metrics": basic_metrics,
            "advanced_metrics": advanced_metrics,
            "summary": asdict(summary),
            "test_prompts": [asdict(tp) for tp in self.test_prompts]
        }
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        logger.info(f"📁 Results saved to {self.output_dir}")
    
    def compare_models(self, model_ids: List[str]) -> None:
        """Compare multiple models and generate comparison report"""
        logger.info(f"🔄 Comparing {len(model_ids)} models...")
        
        comparison_results = []
        
        for model_id in model_ids:
            try:
                logger.info(f"Evaluating {model_id}...")
                
                # Download model
                self.download_model(model_id)
                
                # Run evaluations
                basic_metrics = self.run_basic_evaluation(model_id)
                advanced_metrics = self.run_advanced_evaluation(model_id)
                
                # Generate report
                metrics = self.generate_report(model_id, basic_metrics, advanced_metrics)
                comparison_results.append(metrics)
                
                # Add to results
                self.results.append(metrics)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_id}: {e}")
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results)
    
    def _generate_comparison_report(self, results: List[EvaluationMetrics]) -> None:
        """Generate visual comparison report"""
        if not results:
            return
        
        logger.info("📈 Generating comparison report...")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison Report', fontsize=16)
        
        model_names = [r.model_name.split('/')[-1] for r in results]
        
        # Plot 1: Overall Score
        scores = [r.overall_score for r in results]
        axes[0, 0].bar(model_names, scores, color='skyblue')
        axes[0, 0].set_title('Overall Score')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Inference Time
        times = [r.inference_time for r in results]
        axes[0, 1].bar(model_names, times, color='lightcoral')
        axes[0, 1].set_title('Inference Time (seconds)')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory Usage
        memory = [r.memory_peak_mb for r in results]
        axes[0, 2].bar(model_names, memory, color='lightgreen')
        axes[0, 2].set_title('Memory Usage (MB)')
        axes[0, 2].set_ylabel('Memory (MB)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Quality Metrics
        quality_scores = [r.image_quality_score for r in results]
        adherence_scores = [r.prompt_adherence_score for r in results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, quality_scores, width, label='Image Quality', color='gold')
        axes[1, 0].bar(x + width/2, adherence_scores, width, label='Prompt Adherence', color='orange')
        axes[1, 0].set_title('Quality Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        
        # Plot 5: Style & Aesthetic
        style_scores = [r.style_consistency_score for r in results]
        aesthetic_scores = [r.aesthetic_score for r in results]
        
        axes[1, 1].bar(x - width/2, style_scores, width, label='Style Consistency', color='purple')
        axes[1, 1].bar(x + width/2, aesthetic_scores, width, label='Aesthetic Quality', color='pink')
        axes[1, 1].set_title('Style & Aesthetic')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].legend()
        
        # Plot 6: Diversity vs Technical
        diversity_scores = [r.diversity_score for r in results]
        technical_scores = [r.technical_score for r in results]
        
        axes[1, 2].bar(x - width/2, diversity_scores, width, label='Diversity', color='cyan')
        axes[1, 2].bar(x + width/2, technical_scores, width, label='Technical Quality', color='magenta')
        axes[1, 2].set_title('Diversity vs Technical')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(model_names, rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'model_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text summary
        self._generate_text_summary(results)
        
        logger.info(f"📊 Comparison report saved to {plot_file}")
    
    def _generate_text_summary(self, results: List[EvaluationMetrics]) -> None:
        """Generate text summary of comparison"""
        summary_file = self.output_dir / 'comparison_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("MODEL EVALUATION COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Sort by overall score
            sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
            
            f.write("RANKING BY OVERALL SCORE:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. {result.model_name}\n")
                f.write(f"   Overall Score: {result.overall_score:.3f}\n")
                f.write(f"   Inference Time: {result.inference_time:.2f}s\n")
                f.write(f"   Memory Usage: {result.memory_peak_mb:.1f} MB\n")
                f.write(f"   Quality Score: {result.image_quality_score:.3f}\n")
                f.write(f"   Prompt Adherence: {result.prompt_adherence_score:.3f}\n")
                f.write(f"   Style Consistency: {result.style_consistency_score:.3f}\n")
                f.write(f"   Aesthetic Quality: {result.aesthetic_score:.3f}\n")
                f.write(f"   Technical Quality: {result.technical_score:.3f}\n")
                f.write(f"   Diversity Score: {result.diversity_score:.3f}\n\n")
            
            # Best in each category
            f.write("BEST IN EACH CATEGORY:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Fastest: {min(results, key=lambda x: x.inference_time).model_name}\n")
            f.write(f"Most Efficient: {min(results, key=lambda x: x.memory_peak_mb).model_name}\n")
            f.write(f"Best Quality: {max(results, key=lambda x: x.image_quality_score).model_name}\n")
            f.write(f"Most Adherent: {max(results, key=lambda x: x.prompt_adherence_score).model_name}\n")
            f.write(f"Most Consistent: {max(results, key=lambda x: x.style_consistency_score).model_name}\n")
            f.write(f"Most Aesthetic: {max(results, key=lambda x: x.aesthetic_score).model_name}\n")
            f.write(f"Most Diverse: {max(results, key=lambda x: x.diversity_score).model_name}\n")
        
        logger.info(f"📝 Text summary saved to {summary_file}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Comprehensive Diffusion Model Evaluator")
    parser.add_argument("--model", type=str, help="Model ID to evaluate (e.g., 'runwayml/stable-diffusion-v1-5')")
    parser.add_argument("--models", nargs="+", help="Multiple model IDs for comparison")
    parser.add_argument("--eval-type", choices=["basic", "advanced", "all"], default="all", help="Type of evaluation to run")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples for evaluation")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for model downloads")
    
    args = parser.parse_args()
    
    if not args.model and not args.models:
        logger.error("Please specify either --model or --models")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    
    try:
        if args.models:
            # Compare multiple models
            logger.info(f"🔄 Comparing {len(args.models)} models...")
            evaluator.compare_models(args.models)
        else:
            # Evaluate single model
            logger.info(f"🔍 Evaluating single model: {args.model}")
            
            # Download model
            evaluator.download_model(args.model, args.cache_dir)
            
            # Run evaluations
            basic_metrics = None
            advanced_metrics = None
            
            if args.eval_type in ["basic", "all"]:
                basic_metrics = evaluator.run_basic_evaluation(args.model, args.num_samples)
            
            if args.eval_type in ["advanced", "all"]:
                advanced_metrics = evaluator.run_advanced_evaluation(args.model, args.num_samples)
            
            # Generate report
            if basic_metrics or advanced_metrics:
                metrics = evaluator.generate_report(args.model, basic_metrics or {}, advanced_metrics or {})
                evaluator.results.append(metrics)
                
                logger.info("🎉 Evaluation completed successfully!")
                logger.info(f"📊 Overall Score: {metrics.overall_score:.3f}")
                logger.info(f"⏱️  Average Inference Time: {metrics.inference_time:.2f}s")
                logger.info(f"💾 Memory Usage: {metrics.memory_peak_mb:.1f} MB")
                logger.info(f"🎨 Quality Score: {metrics.image_quality_score:.3f}")
                logger.info(f"🎯 Prompt Adherence: {metrics.prompt_adherence_score:.3f}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
