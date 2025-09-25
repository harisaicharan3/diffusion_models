#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

def test_imports():
    """Test all imports."""
    print("🧪 Testing AI Image Generator Imports")
    print("=" * 40)
    
    try:
        # Test core imports
        print("1. Testing core diffusion models...")
        from diffusion_models import DiffusionModelManager, GenerationConfig, ModelInfo
        print("   ✅ DiffusionModelManager imported")
        print("   ✅ GenerationConfig imported")
        print("   ✅ ModelInfo imported")
        
        # Test image processor
        print("2. Testing image processor...")
        from image_processor import ImageProcessor
        print("   ✅ ImageProcessor imported")
        
        # Test prompt examples
        print("3. Testing prompt examples...")
        from examples.prompt_examples import PromptExamples
        print("   ✅ PromptExamples imported")
        
        # Test model manager initialization
        print("4. Testing model manager initialization...")
        manager = DiffusionModelManager()
        print("   ✅ DiffusionModelManager initialized")
        
        # Test available models
        models = manager.get_available_models()
        print(f"   ✅ Found {len(models)} available models")
        
        # Test device info
        device_info = manager.get_device_info()
        print(f"   ✅ Device: {device_info['device']}")
        print(f"   ✅ CUDA Available: {device_info['cuda_available']}")
        
        # Test image processor initialization
        print("5. Testing image processor initialization...")
        processor = ImageProcessor()
        print("   ✅ ImageProcessor initialized")
        
        # Test prompt examples
        print("6. Testing prompt examples...")
        examples = PromptExamples()
        categories = examples.get_categories()
        print(f"   ✅ Found {len(categories)} prompt categories")
        
        print("\n🎉 All imports and initializations successful!")
        print("\nTo run the Streamlit app:")
        print("streamlit run image_generator.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
