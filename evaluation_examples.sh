#!/bin/bash

# Model Evaluation Examples
# This script demonstrates various ways to use the evaluation system

echo "🧪 Model Evaluation Examples"
echo "============================"
echo ""

# Check if we're in the right directory
if [ ! -f "model_evaluator.py" ]; then
    echo "❌ Please run this script from the diffusion_models directory"
    exit 1
fi

echo "📋 Available evaluation commands:"
echo ""

echo "1. 🚀 Quick Test (Fast models, basic prompts)"
echo "   python run_evaluation.py --config quick_test"
echo ""

echo "2. 📊 Comprehensive Evaluation (Popular models, diverse prompts)"
echo "   python run_evaluation.py --config comprehensive"
echo ""

echo "3. 🎨 Artistic Focus (Creative models, artistic prompts)"
echo "   python run_evaluation.py --config artistic_focus"
echo ""

echo "4. ⚡ Speed Benchmark (Fastest models, performance focus)"
echo "   python run_evaluation.py --config speed_benchmark"
echo ""

echo "5. 🏆 Quality Benchmark (High-quality models, quality focus)"
echo "   python run_evaluation.py --config quality_benchmark"
echo ""

echo "6. 🔍 Custom Single Model Evaluation"
echo "   python run_evaluation.py --custom --model 'runwayml/stable-diffusion-v1-5' --eval-type basic"
echo ""

echo "7. 🔬 Advanced Custom Evaluation"
echo "   python run_evaluation.py --custom --model 'stabilityai/stable-diffusion-xl-base-1.0' --eval-type advanced"
echo ""

echo "8. 🎯 Complete Custom Evaluation"
echo "   python run_evaluation.py --custom --model 'kandinsky-community/kandinsky-2-2-decoder' --eval-type all"
echo ""

echo "9. 📈 Model Comparison (Override models in config)"
echo "   python run_evaluation.py --config quick_test --models 'runwayml/stable-diffusion-v1-5' 'stabilityai/stable-diffusion-2-1'"
echo ""

echo "10. 📁 Custom Output Directory"
echo "    python run_evaluation.py --config comprehensive --output-dir results/my_comparison"
echo ""

echo "11. 📋 List All Available Configurations"
echo "    python run_evaluation.py --list-configs"
echo ""

echo "12. 🎬 Run Demo (Interactive demonstration)"
echo "    python demo_evaluation.py"
echo ""

echo "📚 Documentation:"
echo "   - EVALUATION_GUIDE.md - Comprehensive guide"
echo "   - DIFFUSION_MODELS.md - Technical explanation"
echo "   - README.md - Main project documentation"
echo ""

echo "⚙️  Prerequisites:"
echo "   - Install evaluation dependencies: pip install -r requirements_eval.txt"
echo "   - Ensure main dependencies are installed: pip install -r requirements.txt"
echo "   - Have sufficient disk space for model downloads (~4GB per model)"
echo ""

echo "💡 Tips:"
echo "   - Start with 'quick_test' for a fast overview"
echo "   - Use 'speed_benchmark' to compare performance"
echo "   - Use 'quality_benchmark' for best quality models"
echo "   - Results are saved in 'evaluation_results/' by default"
echo ""

echo "🎯 Ready to evaluate! Choose a command above and run it."
