#!/bin/bash

echo "🎨 AI Image Generator Setup"
echo "=========================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not available. Please install pip."
    exit 1
fi

echo "✅ pip3 found"

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --user -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo "💡 Try running: xcode-select --install"
    echo "   Then run this script again"
    exit 1
fi

# Check for CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA Version: {torch.version.cuda}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
else:
    print('⚠️  CUDA not available - will use CPU (slower)')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the application:"
echo "streamlit run image_generator.py"
echo ""
echo "To run the demo:"
echo "python3 examples/demo.py"
echo ""
echo "💡 Tips:"
echo "- For best performance, use a GPU with 6GB+ VRAM"
echo "- Start with Stable Diffusion 1.5 for faster generation"
echo "- Use memory-efficient mode if you have limited VRAM"
