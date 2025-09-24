"""
Image processing utilities for post-processing generated images.
Includes upscaling, filtering, enhancement, and format conversion.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Optional, Dict, Any
import io
import base64

class ImageProcessor:
    """Handles image processing and enhancement operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_formats = ['PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF']
    
    def upscale_image(self, image: Image.Image, scale_factor: float = 2.0, method: str = 'lanczos') -> Image.Image:
        """
        Upscale an image using various methods.
        
        Args:
            image: Input PIL Image
            scale_factor: Scaling factor (2.0 = 2x, 4.0 = 4x)
            method: Upscaling method ('lanczos', 'bicubic', 'nearest')
            
        Returns:
            Upscaled PIL Image
        """
        if scale_factor <= 1.0:
            return image
        
        # Calculate new dimensions
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Choose resampling method
        resample_methods = {
            'lanczos': Image.Resampling.LANCZOS,
            'bicubic': Image.Resampling.BICUBIC,
            'nearest': Image.Resampling.NEAREST
        }
        
        resample = resample_methods.get(method, Image.Resampling.LANCZOS)
        
        # Upscale the image
        upscaled = image.resize((new_width, new_height), resample=resample)
        
        return upscaled
    
    def enhance_image(self, image: Image.Image, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> Image.Image:
        """
        Enhance image with brightness, contrast, saturation, and sharpness adjustments.
        
        Args:
            image: Input PIL Image
            brightness: Brightness multiplier (1.0 = no change)
            contrast: Contrast multiplier (1.0 = no change)
            saturation: Saturation multiplier (1.0 = no change)
            sharpness: Sharpness multiplier (1.0 = no change)
            
        Returns:
            Enhanced PIL Image
        """
        enhanced = image.copy()
        
        # Apply brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)
        
        # Apply contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
        
        # Apply saturation
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)
        
        # Apply sharpness
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
        
        return enhanced
    
    def apply_filter(self, image: Image.Image, filter_type: str) -> Image.Image:
        """
        Apply various filters to the image.
        
        Args:
            image: Input PIL Image
            filter_type: Type of filter to apply
            
        Returns:
            Filtered PIL Image
        """
        filtered = image.copy()
        
        filters = {
            'blur': ImageFilter.BLUR,
            'smooth': ImageFilter.SMOOTH,
            'sharpen': ImageFilter.SHARPEN,
            'edge_enhance': ImageFilter.EDGE_ENHANCE,
            'emboss': ImageFilter.EMBOSS,
            'contour': ImageFilter.CONTOUR,
            'detail': ImageFilter.DETAIL,
            'smooth_more': ImageFilter.SMOOTH_MORE,
            'sharpen_more': ImageFilter.SHARPEN_MORE,
            'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE
        }
        
        if filter_type in filters:
            filtered = filtered.filter(filters[filter_type])
        
        return filtered
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background from image (simple implementation).
        Note: This is a basic implementation. For production use, consider using
        specialized background removal libraries like rembg.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Image with transparent background
        """
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple background removal based on edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create mask
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Apply mask to alpha channel
        img_array[:, :, 3] = mask
        
        return Image.fromarray(img_array, 'RGBA')
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Create a thumbnail of the image.
        
        Args:
            image: Input PIL Image
            size: Thumbnail size (width, height)
            
        Returns:
            Thumbnail PIL Image
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    def create_grid(self, images: List[Image.Image], 
                   grid_size: Optional[Tuple[int, int]] = None,
                   spacing: int = 10) -> Image.Image:
        """
        Create a grid layout of multiple images.
        
        Args:
            images: List of PIL Images
            grid_size: Grid dimensions (rows, cols). If None, auto-calculate
            spacing: Spacing between images in pixels
            
        Returns:
            Grid PIL Image
        """
        if not images:
            raise ValueError("No images provided")
        
        # Calculate grid size if not provided
        if grid_size is None:
            num_images = len(images)
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Get image dimensions (assume all images are the same size)
        img_width, img_height = images[0].size
        
        # Calculate grid dimensions
        grid_width = cols * img_width + (cols - 1) * spacing
        grid_height = rows * img_height + (rows - 1) * spacing
        
        # Create grid image
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # Place images in grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            x = col * (img_width + spacing)
            y = row * (img_height + spacing)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    def add_watermark(self, image: Image.Image, 
                     text: str = "AI Generated",
                     position: str = "bottom-right",
                     opacity: float = 0.7) -> Image.Image:
        """
        Add a text watermark to the image.
        
        Args:
            image: Input PIL Image
            text: Watermark text
            position: Position of watermark ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            opacity: Watermark opacity (0.0 to 1.0)
            
        Returns:
            Image with watermark
        """
        from PIL import ImageDraw, ImageFont
        
        watermarked = image.copy()
        
        # Create a transparent overlay
        overlay = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        positions = {
            'top-left': (10, 10),
            'top-right': (watermarked.width - text_width - 10, 10),
            'bottom-left': (10, watermarked.height - text_height - 10),
            'bottom-right': (watermarked.width - text_width - 10, watermarked.height - text_height - 10)
        }
        
        x, y = positions.get(position, positions['bottom-right'])
        
        # Draw text with opacity
        alpha = int(255 * opacity)
        draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))
        
        # Composite overlay onto image
        if watermarked.mode != 'RGBA':
            watermarked = watermarked.convert('RGBA')
        
        watermarked = Image.alpha_composite(watermarked, overlay)
        
        return watermarked
    
    def convert_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: Input PIL Image
            format: Output format
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def save_image(self, image: Image.Image, 
                  filepath: str, 
                  format: str = 'PNG',
                  quality: int = 95) -> bool:
        """
        Save image to file.
        
        Args:
            image: Input PIL Image
            filepath: Output file path
            format: Output format
            quality: JPEG quality (1-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_kwargs = {}
            if format.upper() == 'JPEG':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            
            image.save(filepath, format=format, **save_kwargs)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get information about an image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary with image information
        """
        return {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown'),
            'size_bytes': len(image.tobytes()),
            'aspect_ratio': image.width / image.height
        }
    
    def batch_process(self, images: List[Image.Image], 
                     operations: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Apply multiple operations to a batch of images.
        
        Args:
            images: List of input PIL Images
            operations: List of operation dictionaries
            
        Returns:
            List of processed PIL Images
        """
        processed_images = []
        
        for img in images:
            processed = img.copy()
            
            for operation in operations:
                op_type = operation.get('type')
                params = operation.get('params', {})
                
                if op_type == 'enhance':
                    processed = self.enhance_image(processed, **params)
                elif op_type == 'filter':
                    processed = self.apply_filter(processed, params.get('filter_type'))
                elif op_type == 'upscale':
                    processed = self.upscale_image(processed, **params)
                elif op_type == 'watermark':
                    processed = self.add_watermark(processed, **params)
            
            processed_images.append(processed)
        
        return processed_images
