# 🧪 Model Evaluation Guide

This guide explains how to use the comprehensive model evaluation system to benchmark and compare diffusion models from Hugging Face.

## 📋 Overview

The evaluation system provides:
- **Automated model downloading** from Hugging Face
- **Comprehensive metrics** (speed, quality, adherence, diversity)
- **Visual comparison reports** with charts and rankings
- **Predefined test configurations** for different use cases
- **CLIP-based semantic evaluation** for prompt adherence

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
# Install evaluation-specific requirements
pip install -r requirements_eval.txt
```

### 2. Run Quick Test

```bash
# Test with fast models using basic prompts
python run_evaluation.py --config quick_test
```

### 3. View Results

Results are saved to `evaluation_results/` with:
- `*_summary.json` - Overall metrics
- `*_detailed.json` - Detailed results
- `model_comparison.png` - Visual comparison charts
- `comparison_summary.txt` - Text summary with rankings

## 📊 Available Configurations

### Quick Test (`quick_test`)
- **Models**: Fast SD models (SD 1.5, SD Turbo)
- **Prompts**: Basic photorealistic prompts
- **Use case**: Speed benchmarking, quick validation
- **Time**: ~5-10 minutes

```bash
python run_evaluation.py --config quick_test
```

### Comprehensive (`comprehensive`)
- **Models**: Popular SD models (1.5, 2.1, XL)
- **Prompts**: Basic + complex prompts
- **Use case**: Full model comparison
- **Time**: ~30-60 minutes

```bash
python run_evaluation.py --config comprehensive
```

### Artistic Focus (`artistic_focus`)
- **Models**: Artistic models (Kandinsky, SD XL)
- **Prompts**: Creative and artistic prompts
- **Use case**: Creative applications
- **Time**: ~45-90 minutes

```bash
python run_evaluation.py --config artistic_focus
```

### Speed Benchmark (`speed_benchmark`)
- **Models**: Fastest models available
- **Prompts**: Basic prompts only
- **Use case**: Performance comparison
- **Time**: ~10-20 minutes

```bash
python run_evaluation.py --config speed_benchmark
```

### Quality Benchmark (`quality_benchmark`)
- **Models**: High-quality models (SD XL, SD 2.1)
- **Prompts**: Complex and artistic prompts
- **Use case**: Quality-focused evaluation
- **Time**: ~60-120 minutes

```bash
python run_evaluation.py --config quality_benchmark
```

## 🎯 Custom Evaluations

### Single Model Evaluation

```bash
# Basic evaluation (speed, memory, basic quality)
python run_evaluation.py --custom --model "runwayml/stable-diffusion-v1-5" --eval-type basic

# Advanced evaluation (semantic, style, diversity)
python run_evaluation.py --custom --model "stabilityai/stable-diffusion-xl-base-1.0" --eval-type advanced

# Complete evaluation (both basic and advanced)
python run_evaluation.py --custom --model "kandinsky-community/kandinsky-2-2-decoder" --eval-type all
```

### Override Models in Predefined Config

```bash
# Use artistic config but with different models
python run_evaluation.py --config artistic_focus --models "runwayml/stable-diffusion-v1-5" "stabilityai/stable-diffusion-2-1"
```

### Custom Output Directory

```bash
# Save results to custom directory
python run_evaluation.py --config comprehensive --output-dir results/my_comparison
```

## 📈 Understanding Results

### Metrics Explained

#### Basic Metrics
- **Inference Time**: Average time to generate one image
- **Memory Usage**: Peak RAM/VRAM usage during generation
- **Image Quality**: Technical quality (sharpness, contrast, brightness)
- **Success Rate**: Percentage of successful generations

#### Advanced Metrics
- **Prompt Adherence**: How well images match the text prompt (CLIP similarity)
- **Style Consistency**: Consistency across multiple generations
- **Diversity Score**: Variety in generated images
- **Aesthetic Quality**: Visual appeal and composition
- **Technical Quality**: Image sharpness and artifact levels

#### Overall Score
Weighted combination of all metrics:
- Image Quality: 20%
- Prompt Adherence: 25%
- Style Consistency: 15%
- Diversity: 10%
- Aesthetic Quality: 15%
- Technical Quality: 15%

### Reading the Reports

#### JSON Files
- **Summary**: Overall scores and key metrics
- **Detailed**: Complete results with individual prompt scores

#### PNG Charts
- **Overall Score**: Bar chart comparing models
- **Performance**: Inference time and memory usage
- **Quality Metrics**: Image quality vs prompt adherence
- **Style & Aesthetic**: Style consistency vs aesthetic quality
- **Diversity vs Technical**: Variety vs technical quality

#### Text Summary
- **Ranking**: Models sorted by overall score
- **Best in Category**: Fastest, most efficient, highest quality, etc.
- **Detailed Scores**: All metrics for each model

## 🔧 Advanced Usage

### Using the Low-Level API

```python
from model_evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(output_dir="my_results")

# Download and evaluate models
models = ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"]
evaluator.compare_models(models)

# Access results
for result in evaluator.results:
    print(f"Model: {result.model_name}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Inference Time: {result.inference_time:.2f}s")
```

### Custom Test Prompts

```python
from model_evaluator import TestPrompt
from eval_config import create_custom_config

# Create custom prompts
custom_prompts = [
    TestPrompt(
        prompt="a futuristic robot in a cyberpunk city",
        category="custom",
        difficulty="medium",
        expected_style="cyberpunk",
        key_elements=["robot", "cyberpunk", "city"]
    )
]

# Create custom configuration
config = create_custom_config(
    models=["runwayml/stable-diffusion-v1-5"],
    prompts=[asdict(p) for p in custom_prompts],
    eval_params={"num_inference_steps": 25, "guidance_scale": 8.0},
    output_settings={"save_images": True, "generate_plots": True}
)
```

## 🎨 Supported Models

### Stable Diffusion Models
- `runwayml/stable-diffusion-v1-5` - Fast, reliable
- `stabilityai/stable-diffusion-2-1` - Improved quality
- `stabilityai/stable-diffusion-xl-base-1.0` - High resolution
- `stabilityai/sd-turbo` - Very fast generation

### Other Models
- `kandinsky-community/kandinsky-2-2-decoder` - Artistic style
- `CompVis/stable-diffusion-v-1-4-original` - Original SD

### Adding New Models
Simply use any Hugging Face model ID that's compatible with Diffusers:
```bash
python run_evaluation.py --custom --model "your-model-id" --eval-type all
```

## ⚙️ Performance Tips

### Hardware Optimization
- **GPU**: Use CUDA for faster inference (10-50x speedup)
- **Memory**: Close other applications to free VRAM
- **Storage**: Use SSD for faster model loading

### Evaluation Optimization
- **Quick Tests**: Use `quick_test` config for fast validation
- **Sample Count**: Reduce `--num-samples` for faster evaluation
- **Image Size**: Smaller images = faster generation
- **Steps**: Fewer inference steps = faster but lower quality

### Memory Management
```bash
# For limited VRAM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use CPU offloading
python run_evaluation.py --config quick_test --eval-type basic
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Use smaller models or CPU
python run_evaluation.py --config quick_test --eval-type basic
```

#### Model Download Failures
```bash
# Use custom cache directory
python run_evaluation.py --config quick_test --cache-dir /path/to/cache
```

#### CLIP Loading Issues
```bash
# Evaluation will work without CLIP (uses default scores)
# For CLIP: pip install transformers>=4.30.0
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Example Workflows

### 1. Model Selection Workflow
```bash
# 1. Quick speed test
python run_evaluation.py --config speed_benchmark

# 2. Quality comparison of top performers
python run_evaluation.py --config quality_benchmark --models "model1" "model2"

# 3. Final comprehensive evaluation
python run_evaluation.py --config comprehensive
```

### 2. Custom Model Testing
```bash
# 1. Test new model
python run_evaluation.py --custom --model "new-model-id" --eval-type basic

# 2. Compare with known good model
python run_evaluation.py --config quick_test --models "new-model-id" "runwayml/stable-diffusion-v1-5"
```

### 3. Production Model Selection
```bash
# 1. Speed requirements
python run_evaluation.py --config speed_benchmark

# 2. Quality requirements  
python run_evaluation.py --config quality_benchmark

# 3. Final candidates
python run_evaluation.py --config comprehensive --models "final-candidate-1" "final-candidate-2"
```

## 📚 Further Reading

- [DIFFUSION_MODELS.md](DIFFUSION_MODELS.md) - How diffusion models work
- [README.md](README.md) - Main application guide
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) - Official documentation
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Semantic evaluation method

---

**Happy Evaluating! 🎨📊**
