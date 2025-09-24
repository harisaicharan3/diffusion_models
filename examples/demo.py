#!/usr/bin/env python3
"""
Demo script for the AI Image Generator.
Shows how to use the diffusion models programmatically.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from diffusion_models import DiffusionModelManager, GenerationConfig
from image_processor import ImageProcessor
from prompt_examples import PromptExamples

def demo_basic_generation():
    """Demonstrate basic image generation."""
    print("🎨 AI Image Generator - Basic Demo")
    print("=" * 40)
    
    # Initialize components
    model_manager = DiffusionModelManager()
    image_processor = ImageProcessor()
    
    # Check device
    device_info = model_manager.get_device_info()
    print(f"💻 Device: {device_info['device']}")
    print(f"🔧 CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"🎮 GPU: {device_info['gpu_name']}")
        print(f"💾 GPU Memory: {device_info['gpu_memory_total'] / (1024**3):.1f} GB")
    
    # Load a model
    print("\n🤖 Loading model...")
    success = model_manager.load_model("stable-diffusion-1.5", memory_efficient=True)
    
    if not success:
        print("❌ Failed to load model. Make sure you have enough GPU memory.")
        return
    
    print("✅ Model loaded successfully!")
    
    # Generate an image
    print("\n🎨 Generating image...")
    config = GenerationConfig(
        prompt="A beautiful mountain landscape at sunset with golden light",
        negative_prompt="blurry, low quality, distorted",
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        num_images_per_prompt=1
    )
    
    try:
        images, gen_info = model_manager.generate_images(config)
        
        print(f"✅ Generated {len(images)} image(s) in {gen_info['generation_time']:.2f}s")
        print(f"📊 Model: {gen_info['model']}")
        print(f"📏 Size: {gen_info['width']}x{gen_info['height']}")
        print(f"🔢 Steps: {gen_info['steps']}")
        
        # Save the image
        if images:
            output_path = "demo_generated_image.png"
            images[0].save(output_path)
            print(f"💾 Image saved to: {output_path}")
            
            # Get image info
            img_info = image_processor.get_image_info(images[0])
            print(f"📊 Image info: {img_info['width']}x{img_info['height']}, {img_info['mode']}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

def demo_prompt_examples():
    """Demonstrate prompt examples."""
    print("\n📚 Prompt Examples Demo")
    print("=" * 30)
    
    examples = PromptExamples()
    
    # Show categories
    print("Available categories:")
    for category in examples.get_categories():
        print(f"  • {category}")
    
    # Show examples from each category
    for category in examples.get_categories():
        print(f"\n{category.title()} Examples:")
        category_examples = examples.get_examples_by_category(category)
        
        for i, example in enumerate(category_examples[:2]):  # Show first 2
            print(f"  {i+1}. {example.prompt[:60]}...")
            print(f"     Style: {example.style}")
            print(f"     Recommended Steps: {example.recommended_steps}")

def demo_image_processing():
    """Demonstrate image processing capabilities."""
    print("\n🔧 Image Processing Demo")
    print("=" * 30)
    
    image_processor = ImageProcessor()
    
    # Create a simple test image
    from PIL import Image, ImageDraw
    test_image = Image.new('RGB', (256, 256), color='red')
    draw = ImageDraw.Draw(test_image)
    draw.ellipse([50, 50, 200, 200], fill='blue')
    
    print("Test image created: 256x256 red background with blue circle")
    
    # Test enhancements
    enhanced = image_processor.enhance_image(
        test_image,
        brightness=1.2,
        contrast=1.1,
        saturation=1.3
    )
    print("✅ Image enhancement applied")
    
    # Test upscaling
    upscaled = image_processor.upscale_image(test_image, scale_factor=2.0)
    print(f"✅ Image upscaled: {upscaled.size}")
    
    # Test filters
    filtered = image_processor.apply_filter(test_image, 'blur')
    print("✅ Filter applied")
    
    # Test thumbnail
    thumbnail = image_processor.create_thumbnail(test_image, (64, 64))
    print(f"✅ Thumbnail created: {thumbnail.size}")
    
    print("All image processing functions working correctly!")

def demo_model_comparison():
    """Demonstrate different models."""
    print("\n🤖 Model Comparison Demo")
    print("=" * 30)
    
    model_manager = DiffusionModelManager()
    available_models = model_manager.get_available_models()
    
    print("Available models:")
    for key, info in available_models.items():
        print(f"  • {info.name}")
        print(f"    Description: {info.description}")
        print(f"    Max Resolution: {info.max_resolution}px")
        print(f"    Memory Usage: {info.memory_usage}")
        print(f"    Recommended Steps: {info.recommended_steps}")
        print()

def demo_memory_estimation():
    """Demonstrate memory usage estimation."""
    print("\n💾 Memory Estimation Demo")
    print("=" * 30)
    
    model_manager = DiffusionModelManager()
    
    # Load a model
    success = model_manager.load_model("stable-diffusion-1.5")
    if not success:
        print("❌ Could not load model for memory estimation")
        return
    
    # Test different configurations
    configs = [
        GenerationConfig(width=512, height=512, num_images_per_prompt=1),
        GenerationConfig(width=768, height=768, num_images_per_prompt=1),
        GenerationConfig(width=512, height=512, num_images_per_prompt=4),
        GenerationConfig(width=1024, height=1024, num_images_per_prompt=1)
    ]
    
    for i, config in enumerate(configs):
        memory_info = model_manager.estimate_memory_usage(config)
        print(f"Configuration {i+1}: {config.width}x{config.height}, {config.num_images_per_prompt} images")
        print(f"  Estimated Memory: {memory_info['total_estimated']:.1f} GB")
        print(f"  Recommended GPU: {memory_info['recommended_gpu_memory']} GB")
        print()

def main():
    """Run all demos."""
    print("🚀 AI Image Generator - Complete Demo")
    print("=" * 50)
    
    try:
        demo_basic_generation()
        demo_prompt_examples()
        demo_image_processing()
        demo_model_comparison()
        demo_memory_estimation()
        
        print("\n🎉 All demos completed successfully!")
        print("\nTo run the full Streamlit app:")
        print("streamlit run image_generator.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
