#!/bin/bash

echo "🎨 Starting AI Image Generator"
echo "=============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python3 -c "import streamlit, torch, diffusers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies OK"

# Test imports
echo "🧪 Testing imports..."
python3 test_imports.py
if [ $? -ne 0 ]; then
    echo "❌ Import test failed!"
    exit 1
fi

echo "✅ All tests passed!"

# Start Streamlit app
echo "🚀 Starting Streamlit app..."
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run image_generator.py
