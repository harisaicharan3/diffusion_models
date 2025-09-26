"""
Evaluation Configuration

Configuration file for model evaluation with predefined model sets,
test prompts, and evaluation parameters.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    models: List[str]
    test_prompts: List[Dict[str, Any]]
    evaluation_params: Dict[str, Any]
    output_settings: Dict[str, Any]

# Predefined model sets
POPULAR_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1", 
    "stabilityai/stable-diffusion-xl-base-1.0",
    "kandinsky-community/kandinsky-2-2-decoder"
]

FAST_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/sd-turbo"
]

HIGH_QUALITY_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1"
]

ARTISTIC_MODELS = [
    "kandinsky-community/kandinsky-2-2-decoder",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

# Comprehensive test prompts by category
BASIC_PROMPTS = [
    {
        "prompt": "a red apple on a white table",
        "category": "basic",
        "difficulty": "easy",
        "expected_style": "photorealistic",
        "key_elements": ["apple", "red", "white table"]
    },
    {
        "prompt": "a beautiful sunset over mountains",
        "category": "basic", 
        "difficulty": "easy",
        "expected_style": "landscape",
        "key_elements": ["sunset", "mountains"]
    },
    {
        "prompt": "a cute golden retriever puppy playing in a garden",
        "category": "basic",
        "difficulty": "easy", 
        "expected_style": "photorealistic",
        "key_elements": ["puppy", "golden retriever", "garden"]
    }
]

COMPLEX_PROMPTS = [
    {
        "prompt": "a futuristic city skyline at night with flying cars and neon lights",
        "category": "complex",
        "difficulty": "medium",
        "expected_style": "futuristic",
        "key_elements": ["city", "skyline", "flying cars", "neon lights"]
    },
    {
        "prompt": "a medieval knight in shining armor riding a dragon",
        "category": "fantasy",
        "difficulty": "medium",
        "expected_style": "fantasy art",
        "key_elements": ["knight", "armor", "dragon"]
    },
    {
        "prompt": "a vintage 1950s diner with chrome details and checkered floors",
        "category": "complex",
        "difficulty": "medium",
        "expected_style": "vintage",
        "key_elements": ["diner", "1950s", "chrome", "checkered"]
    }
]

ARTISTIC_PROMPTS = [
    {
        "prompt": "a surreal painting in the style of Salvador Dali featuring melting clocks and floating elephants",
        "category": "artistic",
        "difficulty": "hard",
        "expected_style": "surreal",
        "key_elements": ["melting clocks", "elephants", "Dali style"]
    },
    {
        "prompt": "a hyper-detailed steampunk laboratory with brass gears, copper pipes, and Victorian machinery",
        "category": "complex",
        "difficulty": "hard",
        "expected_style": "steampunk",
        "key_elements": ["steampunk", "brass gears", "copper pipes", "Victorian"]
    },
    {
        "prompt": "a photorealistic portrait of an elderly woman with deep wrinkles and kind eyes, oil painting style",
        "category": "portrait",
        "difficulty": "hard",
        "expected_style": "realistic",
        "key_elements": ["portrait", "elderly woman", "wrinkles", "oil painting"]
    }
]

# Evaluation parameter presets
BASIC_EVAL_PARAMS = {
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "num_samples": 3
}

DETAILED_EVAL_PARAMS = {
    "num_inference_steps": 30,
    "guidance_scale": 8.0,
    "width": 512,
    "height": 512,
    "num_samples": 5
}

HIGH_QUALITY_EVAL_PARAMS = {
    "num_inference_steps": 50,
    "guidance_scale": 9.0,
    "width": 768,
    "height": 768,
    "num_samples": 3
}

# Output settings
STANDARD_OUTPUT = {
    "save_images": True,
    "save_metrics": True,
    "generate_plots": True,
    "generate_report": True,
    "image_format": "PNG",
    "plot_dpi": 300
}

MINIMAL_OUTPUT = {
    "save_images": False,
    "save_metrics": True,
    "generate_plots": False,
    "generate_report": True,
    "image_format": "JPEG",
    "plot_dpi": 150
}

# Predefined configurations
CONFIGURATIONS = {
    "quick_test": EvaluationConfig(
        models=FAST_MODELS[:2],
        test_prompts=BASIC_PROMPTS,
        evaluation_params=BASIC_EVAL_PARAMS,
        output_settings=MINIMAL_OUTPUT
    ),
    
    "comprehensive": EvaluationConfig(
        models=POPULAR_MODELS,
        test_prompts=BASIC_PROMPTS + COMPLEX_PROMPTS,
        evaluation_params=DETAILED_EVAL_PARAMS,
        output_settings=STANDARD_OUTPUT
    ),
    
    "artistic_focus": EvaluationConfig(
        models=ARTISTIC_MODELS,
        test_prompts=ARTISTIC_PROMPTS,
        evaluation_params=HIGH_QUALITY_EVAL_PARAMS,
        output_settings=STANDARD_OUTPUT
    ),
    
    "speed_benchmark": EvaluationConfig(
        models=FAST_MODELS,
        test_prompts=BASIC_PROMPTS,
        evaluation_params=BASIC_EVAL_PARAMS,
        output_settings=MINIMAL_OUTPUT
    ),
    
    "quality_benchmark": EvaluationConfig(
        models=HIGH_QUALITY_MODELS,
        test_prompts=COMPLEX_PROMPTS + ARTISTIC_PROMPTS,
        evaluation_params=HIGH_QUALITY_EVAL_PARAMS,
        output_settings=STANDARD_OUTPUT
    )
}

def get_config(config_name: str) -> EvaluationConfig:
    """Get predefined configuration by name"""
    if config_name not in CONFIGURATIONS:
        available = ", ".join(CONFIGURATIONS.keys())
        raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")
    
    return CONFIGURATIONS[config_name]

def list_configurations() -> List[str]:
    """List all available configurations"""
    return list(CONFIGURATIONS.keys())

def create_custom_config(
    models: List[str],
    prompts: List[Dict[str, Any]],
    eval_params: Dict[str, Any],
    output_settings: Dict[str, Any]
) -> EvaluationConfig:
    """Create custom evaluation configuration"""
    return EvaluationConfig(
        models=models,
        test_prompts=prompts,
        evaluation_params=eval_params,
        output_settings=output_settings
    )
