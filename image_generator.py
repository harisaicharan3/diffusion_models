#!/usr/bin/env python3
"""
AI Image Generator - Streamlit Application
A powerful text-to-image generation app using Hugging Face diffusion models.
"""

import streamlit as st
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Import our modules
from diffusion_models import DiffusionModelManager, GenerationConfig, ModelInfo
from image_processor import ImageProcessor

class ImageGeneratorApp:
    """Main application class for the AI Image Generator."""
    
    def __init__(self):
        """Initialize the application."""
        self.model_manager = DiffusionModelManager()
        self.image_processor = ImageProcessor()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'generated_images' not in st.session_state:
            st.session_state.generated_images = []
        
        if 'generation_history' not in st.session_state:
            st.session_state.generation_history = []
        
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render the sidebar with configuration options."""
        st.sidebar.title("🎨 Image Generator")
        
        # Model selection
        st.sidebar.subheader("🤖 Model Selection")
        available_models = self.model_manager.get_available_models()
        model_names = {info.name: key for key, info in available_models.items()}
        
        selected_model_name = st.sidebar.selectbox(
            "Choose Model",
            list(model_names.keys()),
            index=0,
            help="Select the diffusion model to use for generation"
        )
        
        selected_model_key = model_names[selected_model_name]
        model_info = available_models[selected_model_key]
        
        # Display model info
        with st.sidebar.expander("📋 Model Info"):
            st.write(f"**Description:** {model_info.description}")
            st.write(f"**Max Resolution:** {model_info.max_resolution}px")
            st.write(f"**Memory Usage:** {model_info.memory_usage.title()}")
            st.write(f"**Recommended Steps:** {model_info.recommended_steps}")
        
        # Generation settings
        st.sidebar.subheader("⚙️ Generation Settings")
        
        prompt = st.sidebar.text_area(
            "Prompt",
            value="A beautiful landscape with mountains and a lake at sunset",
            height=100,
            help="Describe the image you want to generate"
        )
        
        negative_prompt = st.sidebar.text_area(
            "Negative Prompt",
            value="blurry, low quality, distorted",
            height=60,
            help="Describe what you don't want in the image"
        )
        
        # Image dimensions
        col1, col2 = st.sidebar.columns(2)
        with col1:
            width = st.sidebar.slider(
                "Width",
                min_value=256,
                max_value=min(1024, model_info.max_resolution),
                value=512,
                step=64,
                help="Image width in pixels"
            )
        with col2:
            height = st.sidebar.slider(
                "Height",
                min_value=256,
                max_value=min(1024, model_info.max_resolution),
                value=512,
                step=64,
                help="Image height in pixels"
            )
        
        # Generation parameters
        num_inference_steps = st.sidebar.slider(
            "Steps",
            min_value=10,
            max_value=50,
            value=model_info.recommended_steps,
            help="Number of denoising steps (higher = better quality, slower)"
        )
        
        guidance_scale = st.sidebar.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="How closely to follow the prompt (higher = more adherence)"
        )
        
        num_images = st.sidebar.slider(
            "Number of Images",
            min_value=1,
            max_value=4,
            value=1,
            help="How many images to generate"
        )
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2**32-1,
                value=0,
                help="Random seed for reproducible results (0 = random)"
            )
            
            safety_checker = st.checkbox(
                "Safety Checker",
                value=True,
                help="Enable content safety filtering"
            )
            
            memory_efficient = st.checkbox(
                "Memory Efficient",
                value=model_info.memory_usage in ["medium", "high"],
                help="Use memory-efficient mode (slower but uses less VRAM)"
            )
        
        return {
            "model_key": selected_model_key,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "seed": seed if seed > 0 else None,
            "safety_checker": safety_checker,
            "memory_efficient": memory_efficient
        }
    
    def render_device_info(self):
        """Render device information."""
        st.sidebar.subheader("💻 Device Info")
        device_info = self.model_manager.get_device_info()
        
        with st.sidebar.expander("Hardware Details"):
            st.write(f"**Device:** {device_info['device']}")
            st.write(f"**CUDA Available:** {device_info['cuda_available']}")
            
            if device_info['cuda_available']:
                st.write(f"**GPU:** {device_info['gpu_name']}")
                st.write(f"**CUDA Version:** {device_info['cuda_version']}")
                
                # Memory usage
                total_mem = device_info['gpu_memory_total'] / (1024**3)
                allocated_mem = device_info['gpu_memory_allocated'] / (1024**3)
                cached_mem = device_info['gpu_memory_cached'] / (1024**3)
                
                st.write(f"**Total GPU Memory:** {total_mem:.1f} GB")
                st.write(f"**Allocated:** {allocated_mem:.1f} GB")
                st.write(f"**Cached:** {cached_mem:.1f} GB")
                
                # Memory usage bar
                usage_percent = (allocated_mem / total_mem) * 100
                st.progress(usage_percent / 100, text=f"GPU Memory: {usage_percent:.1f}%")
    
    def load_model(self, model_key: str, memory_efficient: bool = False) -> bool:
        """Load the selected model."""
        if st.session_state.current_model == model_key and st.session_state.model_loaded:
            return True
        
        with st.spinner(f"Loading {model_key}..."):
            success = self.model_manager.load_model(model_key, memory_efficient)
            
            if success:
                st.session_state.current_model = model_key
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded successfully!")
                return True
            else:
                st.error(f"❌ Failed to load model: {model_key}")
                return False
    
    def generate_images(self, config: Dict[str, Any]) -> List[Any]:
        """Generate images based on configuration."""
        # Load model if needed
        if not self.load_model(config['model_key'], config['memory_efficient']):
            return []
        
        # Create generation config
        gen_config = GenerationConfig(
            prompt=config['prompt'],
            negative_prompt=config['negative_prompt'],
            width=config['width'],
            height=config['height'],
            num_inference_steps=config['num_inference_steps'],
            guidance_scale=config['guidance_scale'],
            num_images_per_prompt=config['num_images'],
            seed=config['seed'],
            safety_checker=config['safety_checker']
        )
        
        # Generate images
        try:
            with st.spinner("Generating images..."):
                images, gen_info = self.model_manager.generate_images(gen_config)
            
            # Store in session state
            st.session_state.generated_images = images
            st.session_state.generation_history.append({
                'timestamp': datetime.now(),
                'config': config,
                'info': gen_info,
                'images_count': len(images)
            })
            
            return images
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            return []
    
    def render_generated_images(self, images: List[Any]):
        """Render the generated images."""
        if not images:
            return
        
        st.subheader("🎨 Generated Images")
        
        # Display images
        if len(images) == 1:
            st.image(images[0], caption="Generated Image", use_column_width=True)
        else:
            # Create grid for multiple images
            cols = st.columns(2)
            for i, img in enumerate(images):
                with cols[i % 2]:
                    st.image(img, caption=f"Image {i+1}", use_column_width=True)
        
        # Image info
        with st.expander("📊 Image Information"):
            for i, img in enumerate(images):
                info = self.image_processor.get_image_info(img)
                st.write(f"**Image {i+1}:**")
                st.write(f"- Size: {info['width']}x{info['height']}")
                st.write(f"- Format: {info['mode']}")
                st.write(f"- Size: {info['size_bytes'] / 1024:.1f} KB")
                st.write(f"- Aspect Ratio: {info['aspect_ratio']:.2f}")
    
    def render_image_processing(self, images: List[Any]):
        """Render image processing options."""
        if not images:
            return
        
        st.subheader("🔧 Image Processing")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Enhancement**")
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
        
        with col2:
            st.write("**Filters**")
            filter_type = st.selectbox(
                "Filter",
                ["none", "blur", "smooth", "sharpen", "edge_enhance", "emboss", "contour"]
            )
            
            st.write("**Upscaling**")
            upscale_factor = st.slider("Scale Factor", 1.0, 4.0, 2.0, 0.5)
        
        # Process images
        if st.button("🔄 Process Images"):
            processed_images = []
            
            for img in images:
                # Apply enhancements
                processed = self.image_processor.enhance_image(
                    img, brightness, contrast, saturation, sharpness
                )
                
                # Apply filter
                if filter_type != "none":
                    processed = self.image_processor.apply_filter(processed, filter_type)
                
                # Upscale
                if upscale_factor > 1.0:
                    processed = self.image_processor.upscale_image(processed, upscale_factor)
                
                processed_images.append(processed)
            
            # Update session state
            st.session_state.generated_images = processed_images
            st.success("✅ Images processed successfully!")
            st.rerun()
    
    def render_export_options(self, images: List[Any]):
        """Render export options."""
        if not images:
            return
        
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download All Images"):
                # Create zip file with all images
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i, img in enumerate(images):
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='PNG')
                        zip_file.writestr(f"generated_image_{i+1}.png", img_bytes.getvalue())
                
                st.download_button(
                    label="Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"generated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        with col2:
            if st.button("📋 Copy to Clipboard"):
                # Convert first image to base64 for clipboard
                if images:
                    img_b64 = self.image_processor.convert_to_base64(images[0])
                    st.code(img_b64, language="text")
                    st.info("Base64 data copied to clipboard")
        
        with col3:
            if st.button("🗑️ Clear Images"):
                st.session_state.generated_images = []
                st.rerun()
    
    def render_generation_history(self):
        """Render generation history."""
        if not st.session_state.generation_history:
            return
        
        st.subheader("📚 Generation History")
        
        for i, entry in enumerate(reversed(st.session_state.generation_history[-5:])):  # Show last 5
            with st.expander(f"Generation {len(st.session_state.generation_history) - i} - {entry['timestamp'].strftime('%H:%M:%S')}"):
                st.write(f"**Prompt:** {entry['config']['prompt']}")
                st.write(f"**Model:** {entry['info']['model']}")
                st.write(f"**Size:** {entry['info']['width']}x{entry['info']['height']}")
                st.write(f"**Steps:** {entry['info']['steps']}")
                st.write(f"**Time:** {entry['info']['generation_time']:.2f}s")
    
    def run(self):
        """Run the main application."""
        # Set page config
        st.set_page_config(
            page_title="AI Image Generator",
            page_icon="🎨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("🎨 AI Image Generator")
        st.markdown("Create stunning images from text descriptions using state-of-the-art diffusion models!")
        
        # Render sidebar
        config = self.render_sidebar()
        self.render_device_info()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Generate button
            if st.button("🚀 Generate Images", type="primary", use_container_width=True):
                images = self.generate_images(config)
                if images:
                    self.render_generated_images(images)
            
            # Display generated images
            if st.session_state.generated_images:
                self.render_generated_images(st.session_state.generated_images)
                self.render_image_processing(st.session_state.generated_images)
                self.render_export_options(st.session_state.generated_images)
        
        with col2:
            # Quick examples
            st.subheader("💡 Example Prompts")
            
            example_prompts = [
                "A majestic mountain landscape at sunset with golden light",
                "Portrait of a wise old wizard with a long white beard",
                "Modern city skyline at night with neon lights",
                "Van Gogh style painting of a starry night",
                "Watercolor painting of a peaceful garden",
                "Digital art of a futuristic robot",
                "Dreams floating in a cosmic void",
                "Music notes dancing in the air"
            ]
            
            for prompt in example_prompts:
                if st.button(prompt, key=f"example_{prompt[:20]}"):
                    st.session_state.prompt = prompt
                    st.rerun()
            
            # Generation history
            self.render_generation_history()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>🎨 AI Image Generator | Built with Hugging Face Diffusers & Streamlit</p>
                <p>Create amazing images with the power of AI!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main entry point."""
    app = ImageGeneratorApp()
    app.run()

if __name__ == "__main__":
    main()
