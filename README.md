# 🎨 AI Image Generator

A powerful text-to-image generation application using Hugging Face's diffusion models!

## ✨ Features

- 🖼️ **Text-to-Image Generation**: Create stunning images from text descriptions
- 🎯 **Multiple Models**: Support for Stable Diffusion, DALL-E, and more
- ⚙️ **Customizable Parameters**: Control image size, quality, and style
- 🎨 **Image Enhancement**: Upscaling, filtering, and post-processing
- 📱 **Beautiful UI**: Clean, intuitive Streamlit interface
- 💾 **Export Options**: Save images in various formats
- 🚀 **Fast Generation**: Optimized for speed and quality

## 🚀 Quick Start

### **Option 1: Automated Setup (Recommended)**
```bash
# 1. Run the setup script
./setup.sh

# 2. Start the application
./run_app.sh
```

### **Option 2: Manual Setup**
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test imports
python3 test_imports.py

# 4. Run the application
streamlit run image_generator.py
```

### **Option 3: Direct Run (if dependencies already installed)**
```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run image_generator.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Supported Models

- **Stable Diffusion 1.5**: Fast, high-quality generation
- **Stable Diffusion 2.1**: Enhanced quality and detail
- **Stable Diffusion XL**: Ultra-high resolution images
- **Kandinsky 2.2**: Artistic and creative styles
- **DALL-E Mini**: Quick and lightweight generation

## 🎨 Example Prompts

### Realistic Images
- "A majestic mountain landscape at sunset with golden light"
- "Portrait of a wise old wizard with a long white beard"
- "Modern city skyline at night with neon lights"

### Artistic Styles
- "Van Gogh style painting of a starry night"
- "Watercolor painting of a peaceful garden"
- "Digital art of a futuristic robot"

### Abstract Concepts
- "Dreams floating in a cosmic void"
- "Music notes dancing in the air"
- "Time flowing like a river"

## ⚙️ Configuration

### Image Settings
- **Width/Height**: 512x512 to 1024x1024 pixels
- **Steps**: 20-50 (higher = better quality, slower)
- **Guidance Scale**: 7.5-15 (higher = more prompt adherence)
- **Seed**: For reproducible results

### Model Settings
- **Model Selection**: Choose from available models
- **Safety Filter**: Enable/disable content filtering
- **Memory Optimization**: For lower-end hardware

## 🛠️ Advanced Features

### Image Processing
- **Upscaling**: 2x, 4x resolution enhancement
- **Style Transfer**: Apply artistic filters
- **Color Correction**: Adjust brightness, contrast, saturation
- **Background Removal**: Isolate subjects

### Batch Generation
- **Multiple Images**: Generate several variations
- **Prompt Variations**: Automatic prompt modifications
- **Grid Layout**: Arrange multiple images

## 📁 Project Structure

```
diffusion_models/
├── image_generator.py      # Main Streamlit application
├── diffusion_models.py     # Core diffusion model logic
├── image_processor.py      # Image processing utilities
├── model_manager.py        # Model loading and management
├── examples/              # Sample prompts and demos
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔧 Hardware Requirements

### Minimum
- **GPU**: 4GB VRAM (GTX 1060, RTX 2060)
- **RAM**: 8GB system memory
- **Storage**: 10GB free space

### Recommended
- **GPU**: 8GB+ VRAM (RTX 3070, RTX 4070)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ free space

## 🚀 Performance Tips

1. **Use appropriate model size** for your hardware
2. **Enable memory optimization** for lower-end GPUs
3. **Start with lower steps** for faster generation
4. **Use smaller image sizes** for quicker results
5. **Close other applications** to free up GPU memory

## 🎨 Creative Tips

1. **Be specific** in your prompts for better results
2. **Use style keywords** (photorealistic, oil painting, etc.)
3. **Experiment with negative prompts** to avoid unwanted elements
4. **Try different seeds** for variety
5. **Combine multiple concepts** for unique images

## 🔒 Privacy & Safety

- **Local Processing**: All generation happens on your machine
- **No Data Collection**: Your prompts and images stay private
- **Content Filtering**: Optional safety checks for inappropriate content
- **Offline Mode**: Works without internet after initial setup

## 🤝 Contributing

Feel free to contribute to this project! Some ideas:
- Add new diffusion models
- Implement image-to-image generation
- Add more image processing features
- Create custom UI themes
- Add batch processing capabilities

## 📄 License

MIT License - feel free to use this project for your own needs!

---

**Built with ❤️ using Hugging Face Diffusers, Streamlit, and PyTorch**
