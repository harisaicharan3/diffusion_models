#!/usr/bin/env python3
"""
Model Evaluation Runner

Easy-to-use script for running model evaluations with predefined configurations.

Usage Examples:
    python run_evaluation.py --config quick_test
    python run_evaluation.py --config comprehensive --output-dir results/comprehensive_test
    python run_evaluation.py --config artistic_focus --models "kandinsky-community/kandinsky-2-2-decoder"
    python run_evaluation.py --custom --model "runwayml/stable-diffusion-v1-5" --eval-type basic
"""

import argparse
import logging
import sys
from pathlib import Path

from model_evaluator import ModelEvaluator
from eval_config import get_config, list_configurations, create_custom_config, BASIC_PROMPTS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_predefined_evaluation(config_name: str, output_dir: str, custom_models: list = None):
    """Run evaluation with predefined configuration"""
    logger.info(f"🚀 Running evaluation with config: {config_name}")
    
    # Get configuration
    config = get_config(config_name)
    
    # Override models if specified
    models_to_evaluate = custom_models if custom_models else config.models
    
    logger.info(f"📋 Models to evaluate: {models_to_evaluate}")
    logger.info(f"📝 Test prompts: {len(config.test_prompts)} prompts")
    logger.info(f"⚙️  Evaluation params: {config.evaluation_params}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Update evaluator's test prompts with config
    from model_evaluator import TestPrompt
    evaluator.test_prompts = [
        TestPrompt(
            prompt=p["prompt"],
            category=p["category"],
            difficulty=p["difficulty"],
            expected_style=p["expected_style"],
            key_elements=p["key_elements"]
        ) for p in config.test_prompts
    ]
    
    try:
        # Run comparison
        evaluator.compare_models(models_to_evaluate)
        
        logger.info("🎉 Evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

def run_custom_evaluation(
    model: str,
    eval_type: str,
    output_dir: str,
    num_samples: int = 5,
    cache_dir: str = None
):
    """Run custom single-model evaluation"""
    logger.info(f"🔍 Running custom evaluation for: {model}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    try:
        # Download model
        evaluator.download_model(model, cache_dir)
        
        # Run evaluations
        basic_metrics = None
        advanced_metrics = None
        
        if eval_type in ["basic", "all"]:
            logger.info("Running basic evaluation...")
            basic_metrics = evaluator.run_basic_evaluation(model, num_samples)
        
        if eval_type in ["advanced", "all"]:
            logger.info("Running advanced evaluation...")
            advanced_metrics = evaluator.run_advanced_evaluation(model, num_samples)
        
        # Generate report
        if basic_metrics or advanced_metrics:
            metrics = evaluator.generate_report(model, basic_metrics or {}, advanced_metrics or {})
            evaluator.results.append(metrics)
            
            logger.info("🎉 Custom evaluation completed!")
            logger.info(f"📊 Overall Score: {metrics.overall_score:.3f}")
            logger.info(f"⏱️  Average Inference Time: {metrics.inference_time:.2f}s")
            logger.info(f"💾 Memory Usage: {metrics.memory_peak_mb:.1f} MB")
            logger.info(f"🎨 Quality Score: {metrics.image_quality_score:.3f}")
            logger.info(f"🎯 Prompt Adherence: {metrics.prompt_adherence_score:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Custom evaluation failed: {e}")
        raise

def main():
    """Main evaluation runner"""
    parser = argparse.ArgumentParser(
        description="Easy-to-use Model Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with fast models
  python run_evaluation.py --config quick_test
  
  # Comprehensive evaluation
  python run_evaluation.py --config comprehensive
  
  # Artistic models evaluation
  python run_evaluation.py --config artistic_focus
  
  # Speed benchmark
  python run_evaluation.py --config speed_benchmark
  
  # Custom single model evaluation
  python run_evaluation.py --custom --model "runwayml/stable-diffusion-v1-5" --eval-type basic
  
  # Override models in predefined config
  python run_evaluation.py --config quick_test --models "stabilityai/stable-diffusion-xl-base-1.0"
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument("--config", type=str, help="Predefined configuration name")
    config_group.add_argument("--custom", action="store_true", help="Run custom evaluation")
    
    # Model selection
    parser.add_argument("--models", nargs="+", help="Override models for predefined config")
    parser.add_argument("--model", type=str, help="Single model for custom evaluation")
    
    # Evaluation options
    parser.add_argument("--eval-type", choices=["basic", "advanced", "all"], default="all", 
                       help="Type of evaluation (custom mode only)")
    parser.add_argument("--num-samples", type=int, default=5, 
                       help="Number of samples for evaluation (custom mode only)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for model downloads")
    
    # Utility options
    parser.add_argument("--list-configs", action="store_true", help="List available configurations")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        configs = list_configurations()
        print("Available configurations:")
        for config in configs:
            print(f"  - {config}")
        return
    
    # Validate arguments
    if not args.list_configs:
        if not args.config and not args.custom:
            logger.error("Please specify either --config or --custom")
            return
            
        if args.config:
            available_configs = list_configurations()
            if args.config not in available_configs:
                logger.error(f"Unknown configuration: {args.config}")
                logger.error(f"Available configurations: {', '.join(available_configs)}")
                return
        
        if args.custom and not args.model:
            logger.error("Custom evaluation requires --model argument")
            return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.config:
            # Run predefined evaluation
            run_predefined_evaluation(
                config_name=args.config,
                output_dir=args.output_dir,
                custom_models=args.models
            )
        else:
            # Run custom evaluation
            run_custom_evaluation(
                model=args.model,
                eval_type=args.eval_type,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                cache_dir=args.cache_dir
            )
    
    except KeyboardInterrupt:
        logger.info("🛑 Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
