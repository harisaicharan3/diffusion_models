#!/usr/bin/env python3
"""
Example prompts and demo functionality for the AI Image Generator.
Contains categorized prompts for different styles and use cases.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PromptExample:
    """Represents a prompt example with metadata."""
    prompt: str
    negative_prompt: str
    style: str
    category: str
    description: str
    recommended_steps: int = 20
    recommended_guidance: float = 7.5

class PromptExamples:
    """Collection of example prompts for different styles and use cases."""
    
    def __init__(self):
        """Initialize with categorized prompt examples."""
        self.examples = self._load_examples()
    
    def _load_examples(self) -> Dict[str, List[PromptExample]]:
        """Load all example prompts."""
        return {
            "realistic": [
                PromptExample(
                    prompt="A majestic mountain landscape at sunset with golden light, snow-capped peaks, crystal clear lake reflection",
                    negative_prompt="blurry, low quality, distorted, cartoon, anime",
                    style="photorealistic",
                    category="landscape",
                    description="Stunning mountain landscape with perfect lighting",
                    recommended_steps=25,
                    recommended_guidance=8.0
                ),
                PromptExample(
                    prompt="Portrait of a wise old wizard with a long white beard, wearing a blue robe, magical staff in hand, detailed facial features",
                    negative_prompt="blurry, low quality, distorted, cartoon, anime, young person",
                    style="portrait",
                    category="character",
                    description="Detailed character portrait with magical elements",
                    recommended_steps=30,
                    recommended_guidance=7.5
                ),
                PromptExample(
                    prompt="Modern city skyline at night with neon lights, skyscrapers, urban atmosphere, cinematic lighting",
                    negative_prompt="blurry, low quality, distorted, cartoon, anime, daytime",
                    style="urban",
                    category="architecture",
                    description="Dramatic urban nightscape with neon lighting",
                    recommended_steps=25,
                    recommended_guidance=8.5
                ),
                PromptExample(
                    prompt="Close-up macro photography of a dewdrop on a flower petal, morning sunlight, shallow depth of field",
                    negative_prompt="blurry, low quality, distorted, cartoon, anime, dark",
                    style="macro",
                    category="nature",
                    description="Intimate nature photography with perfect focus",
                    recommended_steps=20,
                    recommended_guidance=7.0
                )
            ],
            
            "artistic": [
                PromptExample(
                    prompt="Van Gogh style painting of a starry night, swirling sky, vibrant colors, thick brushstrokes, post-impressionist",
                    negative_prompt="photorealistic, smooth, clean, modern, digital art",
                    style="van_gogh",
                    category="painting",
                    description="Classic Van Gogh style with swirling brushstrokes",
                    recommended_steps=30,
                    recommended_guidance=9.0
                ),
                PromptExample(
                    prompt="Watercolor painting of a peaceful garden, soft colors, flowing brushstrokes, ethereal atmosphere",
                    negative_prompt="photorealistic, sharp, digital art, dark colors, harsh lighting",
                    style="watercolor",
                    category="painting",
                    description="Soft watercolor style with gentle colors",
                    recommended_steps=25,
                    recommended_guidance=8.0
                ),
                PromptExample(
                    prompt="Digital art of a futuristic robot, cyberpunk style, neon lights, detailed mechanical parts, sci-fi atmosphere",
                    negative_prompt="blurry, low quality, distorted, cartoon, anime, old fashioned",
                    style="cyberpunk",
                    category="sci_fi",
                    description="High-tech cyberpunk robot design",
                    recommended_steps=30,
                    recommended_guidance=8.5
                ),
                PromptExample(
                    prompt="Oil painting of a medieval castle on a hill, dramatic clouds, renaissance style, rich colors",
                    negative_prompt="modern, digital art, cartoon, anime, bright colors",
                    style="renaissance",
                    category="architecture",
                    description="Classical renaissance painting style",
                    recommended_steps=35,
                    recommended_guidance=9.0
                )
            ],
            
            "abstract": [
                PromptExample(
                    prompt="Dreams floating in a cosmic void, ethereal forms, soft lighting, mystical atmosphere, abstract art",
                    negative_prompt="realistic, concrete objects, harsh lighting, dark colors",
                    style="abstract",
                    category="conceptual",
                    description="Ethereal abstract concept with cosmic elements",
                    recommended_steps=25,
                    recommended_guidance=7.5
                ),
                PromptExample(
                    prompt="Music notes dancing in the air, colorful sound waves, abstract representation of melody, flowing forms",
                    negative_prompt="realistic, concrete objects, dark colors, static",
                    style="abstract",
                    category="music",
                    description="Abstract representation of music and sound",
                    recommended_steps=20,
                    recommended_guidance=7.0
                ),
                PromptExample(
                    prompt="Time flowing like a river, abstract clockwork, golden gears, flowing motion, conceptual art",
                    negative_prompt="realistic, concrete objects, dark colors, static",
                    style="abstract",
                    category="conceptual",
                    description="Abstract concept of time and motion",
                    recommended_steps=25,
                    recommended_guidance=8.0
                )
            ],
            
            "fantasy": [
                PromptExample(
                    prompt="Dragon soaring over a mystical forest, magical creatures, glowing mushrooms, fantasy atmosphere",
                    negative_prompt="realistic, modern, technology, dark colors, scary",
                    style="fantasy",
                    category="creature",
                    description="Epic fantasy scene with magical elements",
                    recommended_steps=30,
                    recommended_guidance=8.0
                ),
                PromptExample(
                    prompt="Floating islands in the sky, waterfalls cascading down, magical crystals, ethereal lighting",
                    negative_prompt="realistic, ground-based, dark colors, scary, modern",
                    style="fantasy",
                    category="landscape",
                    description="Magical floating landscape with crystals",
                    recommended_steps=25,
                    recommended_guidance=7.5
                ),
                PromptExample(
                    prompt="Elven city in the treetops, wooden bridges, glowing lanterns, magical forest, peaceful atmosphere",
                    negative_prompt="realistic, modern, technology, dark colors, scary, ground-based",
                    style="fantasy",
                    category="architecture",
                    description="Peaceful elven architecture in nature",
                    recommended_steps=30,
                    recommended_guidance=8.5
                )
            ],
            
            "sci_fi": [
                PromptExample(
                    prompt="Space station orbiting a distant planet, futuristic architecture, glowing lights, cosmic background",
                    negative_prompt="realistic, old fashioned, dark colors, scary, primitive",
                    style="sci_fi",
                    category="architecture",
                    description="Futuristic space station design",
                    recommended_steps=25,
                    recommended_guidance=8.0
                ),
                PromptExample(
                    prompt="Holographic interface floating in mid-air, neon blue light, futuristic technology, clean design",
                    negative_prompt="realistic, old fashioned, dark colors, scary, primitive",
                    style="sci_fi",
                    category="technology",
                    description="Advanced holographic technology interface",
                    recommended_steps=20,
                    recommended_guidance=7.5
                ),
                PromptExample(
                    prompt="Alien planet with two moons, strange vegetation, otherworldly atmosphere, sci-fi landscape",
                    negative_prompt="realistic, Earth-like, dark colors, scary, primitive",
                    style="sci_fi",
                    category="landscape",
                    description="Exotic alien world with unique features",
                    recommended_steps=30,
                    recommended_guidance=8.5
                )
            ],
            
            "minimalist": [
                PromptExample(
                    prompt="Simple geometric shapes, clean lines, minimal color palette, modern design, white background",
                    negative_prompt="complex, cluttered, dark colors, detailed, realistic",
                    style="minimalist",
                    category="geometric",
                    description="Clean minimalist geometric design",
                    recommended_steps=15,
                    recommended_guidance=6.0
                ),
                PromptExample(
                    prompt="Single tree silhouette against sunset, simple composition, warm colors, peaceful mood",
                    negative_prompt="complex, cluttered, dark colors, detailed, realistic, multiple objects",
                    style="minimalist",
                    category="nature",
                    description="Simple nature scene with clean composition",
                    recommended_steps=20,
                    recommended_guidance=7.0
                )
            ]
        }
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self.examples.keys())
    
    def get_examples_by_category(self, category: str) -> List[PromptExample]:
        """Get examples for a specific category."""
        return self.examples.get(category, [])
    
    def get_random_example(self, category: str = None) -> PromptExample:
        """Get a random example from a category or all categories."""
        import random
        
        if category and category in self.examples:
            examples = self.examples[category]
        else:
            # Flatten all examples
            examples = []
            for cat_examples in self.examples.values():
                examples.extend(cat_examples)
        
        return random.choice(examples)
    
    def search_examples(self, query: str) -> List[PromptExample]:
        """Search examples by query string."""
        query_lower = query.lower()
        results = []
        
        for category_examples in self.examples.values():
            for example in category_examples:
                if (query_lower in example.prompt.lower() or 
                    query_lower in example.description.lower() or
                    query_lower in example.style.lower()):
                    results.append(example)
        
        return results
    
    def get_recommended_settings(self, style: str) -> Dict[str, Any]:
        """Get recommended settings for a specific style."""
        style_settings = {
            "photorealistic": {"steps": 25, "guidance": 8.0, "width": 512, "height": 512},
            "portrait": {"steps": 30, "guidance": 7.5, "width": 512, "height": 768},
            "landscape": {"steps": 25, "guidance": 8.0, "width": 768, "height": 512},
            "artistic": {"steps": 30, "guidance": 9.0, "width": 512, "height": 512},
            "abstract": {"steps": 25, "guidance": 7.5, "width": 512, "height": 512},
            "fantasy": {"steps": 30, "guidance": 8.0, "width": 512, "height": 512},
            "sci_fi": {"steps": 25, "guidance": 8.0, "width": 512, "height": 512},
            "minimalist": {"steps": 15, "guidance": 6.0, "width": 512, "height": 512}
        }
        
        return style_settings.get(style, {"steps": 20, "guidance": 7.5, "width": 512, "height": 512})
    
    def get_style_tips(self) -> Dict[str, List[str]]:
        """Get tips for different art styles."""
        return {
            "photorealistic": [
                "Use specific lighting descriptions (golden hour, studio lighting)",
                "Include camera settings (macro, shallow depth of field)",
                "Add texture details (rough, smooth, metallic)",
                "Specify quality terms (high resolution, detailed, sharp)"
            ],
            "artistic": [
                "Reference famous artists (Van Gogh, Monet, Picasso)",
                "Specify art medium (oil painting, watercolor, digital art)",
                "Include brushstroke descriptions (thick, flowing, delicate)",
                "Add artistic movements (impressionist, surrealist, abstract)"
            ],
            "fantasy": [
                "Use magical elements (glowing, ethereal, mystical)",
                "Include fantasy creatures and settings",
                "Add atmospheric effects (fog, light rays, particles)",
                "Specify magical lighting (crystal glow, magical aura)"
            ],
            "sci_fi": [
                "Include futuristic technology descriptions",
                "Use clean, geometric design elements",
                "Add glowing lights and holographic effects",
                "Specify advanced materials (metallic, translucent, energy)"
            ],
            "abstract": [
                "Focus on emotions and concepts rather than objects",
                "Use flowing, organic shapes",
                "Experiment with color combinations",
                "Include movement and energy descriptions"
            ]
        }

def main():
    """Demo the prompt examples."""
    examples = PromptExamples()
    
    print("🎨 AI Image Generator - Prompt Examples")
    print("=" * 50)
    
    # Show categories
    print("\n📚 Available Categories:")
    for category in examples.get_categories():
        print(f"  • {category.title()}")
    
    # Show random example
    print("\n🎲 Random Example:")
    random_example = examples.get_random_example()
    print(f"  Prompt: {random_example.prompt}")
    print(f"  Style: {random_example.style}")
    print(f"  Category: {random_example.category}")
    print(f"  Description: {random_example.description}")
    
    # Show style tips
    print("\n💡 Style Tips:")
    tips = examples.get_style_tips()
    for style, style_tips in tips.items():
        print(f"\n  {style.title()}:")
        for tip in style_tips:
            print(f"    • {tip}")

if __name__ == "__main__":
    main()
