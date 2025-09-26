#!/usr/bin/env python3
"""
Demo Evaluation Script

This script demonstrates how to use the model evaluation system
with a simple example that runs quickly.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_evaluator import ModelEvaluator
from eval_config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_evaluation():
    """Demo basic evaluation with a single model"""
    logger.info("🎯 Running Demo: Basic Model Evaluation")
    logger.info("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir="demo_results")
    
    # Use a fast, reliable model for demo
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        logger.info(f"📥 Downloading model: {model_id}")
        evaluator.download_model(model_id)
        
        logger.info("🔍 Running basic evaluation...")
        basic_metrics = evaluator.run_basic_evaluation(model_id, num_samples=2)
        
        logger.info("📊 Basic Evaluation Results:")
        logger.info(f"  ⏱️  Average Inference Time: {basic_metrics.get('inference_times', [0])[0]:.2f}s" if basic_metrics.get('inference_times') else "  ⏱️  No timing data")
        logger.info(f"  💾 Average Memory Usage: {basic_metrics.get('memory_usage', [0])[0]:.1f} MB" if basic_metrics.get('memory_usage') else "  💾 No memory data")
        logger.info(f"  🎨 Average Image Quality: {basic_metrics.get('image_qualities', [0])[0]:.3f}" if basic_metrics.get('image_qualities') else "  🎨 No quality data")
        logger.info(f"  ✅ Success Rate: {basic_metrics.get('success_rate', 0):.1%}")
        
        # Generate report
        metrics = evaluator.generate_report(model_id, basic_metrics, {})
        
        logger.info("\n📈 Generated Report:")
        logger.info(f"  📊 Overall Score: {metrics.overall_score:.3f}")
        logger.info(f"  📁 Results saved to: demo_results/")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def demo_configuration_usage():
    """Demo using predefined configurations"""
    logger.info("\n🎯 Running Demo: Configuration Usage")
    logger.info("=" * 50)
    
    try:
        # Get quick test configuration
        config = get_config("quick_test")
        
        logger.info("📋 Quick Test Configuration:")
        logger.info(f"  🤖 Models: {config.models}")
        logger.info(f"  📝 Prompts: {len(config.test_prompts)} test prompts")
        logger.info(f"  ⚙️  Params: {config.evaluation_params}")
        logger.info(f"  📊 Output: {config.output_settings}")
        
        # Show available configurations
        from eval_config import list_configurations
        configs = list_configurations()
        
        logger.info(f"\n📚 Available Configurations: {', '.join(configs)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration demo failed: {e}")
        return False

def demo_custom_evaluation():
    """Demo custom evaluation setup"""
    logger.info("\n🎯 Running Demo: Custom Evaluation Setup")
    logger.info("=" * 50)
    
    try:
        from model_evaluator import TestPrompt
        from eval_config import create_custom_config
        
        # Create custom test prompt
        custom_prompt = TestPrompt(
            prompt="a cute robot playing with a ball",
            category="custom",
            difficulty="easy",
            expected_style="cute",
            key_elements=["robot", "ball", "playing"]
        )
        
        logger.info("📝 Custom Test Prompt:")
        logger.info(f"  Prompt: {custom_prompt.prompt}")
        logger.info(f"  Category: {custom_prompt.category}")
        logger.info(f"  Difficulty: {custom_prompt.difficulty}")
        logger.info(f"  Expected Style: {custom_prompt.expected_style}")
        logger.info(f"  Key Elements: {custom_prompt.key_elements}")
        
        # Create custom configuration
        config = create_custom_config(
            models=["runwayml/stable-diffusion-v1-5"],
            prompts=[{
                "prompt": custom_prompt.prompt,
                "category": custom_prompt.category,
                "difficulty": custom_prompt.difficulty,
                "expected_style": custom_prompt.expected_style,
                "key_elements": custom_prompt.key_elements
            }],
            eval_params={
                "num_inference_steps": 15,
                "guidance_scale": 7.0,
                "width": 512,
                "height": 512,
                "num_samples": 1
            },
            output_settings={
                "save_images": True,
                "save_metrics": True,
                "generate_plots": False,
                "generate_report": True
            }
        )
        
        logger.info("\n⚙️  Custom Configuration Created:")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  Evaluation Params: {config.evaluation_params}")
        logger.info(f"  Output Settings: {config.output_settings}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom evaluation demo failed: {e}")
        return False

def main():
    """Run all demos"""
    logger.info("🚀 Model Evaluation System Demo")
    logger.info("=" * 60)
    logger.info("This demo shows the key features of the evaluation system.")
    logger.info("Note: Actual model evaluation requires downloading models from Hugging Face.")
    logger.info("This demo shows the setup and configuration without downloading models.")
    
    # Run demos
    demos = [
        ("Configuration Usage", demo_configuration_usage),
        ("Custom Evaluation Setup", demo_custom_evaluation),
    ]
    
    # Only run basic evaluation if user wants to (it downloads models)
    try:
        run_basic_eval = input("\n🤔 Do you want to run actual model evaluation? (requires downloading ~4GB model) [y/N]: ").lower().strip()
    except EOFError:
        # Handle case when running in non-interactive mode
        run_basic_eval = 'n'
        logger.info("Running in non-interactive mode, skipping model evaluation")
    
    if run_basic_eval in ['y', 'yes']:
        demos.insert(0, ("Basic Model Evaluation", demo_basic_evaluation))
    
    results = []
    for demo_name, demo_func in demos:
        logger.info(f"\n🎬 Starting: {demo_name}")
        try:
            success = demo_func()
            results.append((demo_name, success))
            if success:
                logger.info(f"✅ {demo_name} completed successfully!")
            else:
                logger.info(f"❌ {demo_name} failed!")
        except Exception as e:
            logger.error(f"❌ {demo_name} failed with exception: {e}")
            results.append((demo_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 DEMO SUMMARY")
    logger.info("=" * 60)
    
    for demo_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {demo_name}: {status}")
    
    successful_demos = sum(1 for _, success in results if success)
    total_demos = len(results)
    
    logger.info(f"\n🎯 Completed {successful_demos}/{total_demos} demos successfully")
    
    if successful_demos == total_demos:
        logger.info("🎉 All demos completed successfully!")
        logger.info("\n📚 Next Steps:")
        logger.info("  1. Try: python run_evaluation.py --config quick_test")
        logger.info("  2. Try: python run_evaluation.py --custom --model 'runwayml/stable-diffusion-v1-5' --eval-type basic")
        logger.info("  3. Read: EVALUATION_GUIDE.md for detailed instructions")
    else:
        logger.info("⚠️  Some demos failed. Check the error messages above.")

if __name__ == "__main__":
    main()
