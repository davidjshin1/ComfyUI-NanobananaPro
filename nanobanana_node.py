import torch
import numpy as np
import base64
import requests
import json
from PIL import Image
import io

class NanoBananaProNode:
    """
    ComfyUI Node for Nano Banana Pro (Gemini 3 Pro Image Preview)
    Based on Google GenAI API documentation.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Enter your Gemini API Key"}),
                "prompt": ("STRING", {"multiline": True, "default": "A photorealistic shot of..."}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "image_input_1": ("IMAGE", {"default": None}), # Optional input image
            },
            "optional": {
                "image_input_2": ("IMAGE", {"default": None}),
                "image_input_3": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana 🍌"

    def tensor_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
            
        # Handle batch dimension if present (take first image)
        if len(image_tensor.shape) > 3:
            image_tensor = image_tensor[0]
            
        # Convert tensor to PIL Image
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_image(self, api_key, prompt, aspect_ratio, resolution, image_input_1=None, image_input_2=None, image_input_3=None):
        if not api_key:
            raise ValueError("API Key is required for Nano Banana Pro.")

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        
        # Prepare content parts
        parts = [{"text": prompt}]
        
        # Add input images if provided
        for img in [image_input_1, image_input_2, image_input_3]:
            if img is not None:
                b64_data = self.tensor_to_base64(img)
                if b64_data:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": b64_data
                        }
                    })

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution
                }
            }
        }

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        # Make the API request
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            generated_images = []
            response_texts = []

            candidates = result.get('candidates', [])
            if not candidates:
                raise ValueError("No candidates returned from API.")

            for part in candidates[0].get('content', {}).get('parts', []):
                if 'text' in part:
                    response_texts.append(part['text'])
                
                if 'inlineData' in part:
                    img_data = base64.b64decode(part['inlineData']['data'])
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Convert PIL image to torch tensor
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np)[None,]
                    generated_images.append(img_tensor)
            
            if not generated_images:
                # If no image generated, return a blank black image
                print("Warning: No image returned, returning blank.")
                blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (blank, "\n".join(response_texts))

            # Concatenate images if multiple returned (batch)
            final_image_batch = torch.cat(generated_images, dim=0)
            final_text = "\n".join(response_texts)
            
            return (final_image_batch, final_text)

        except Exception as e:
            print(f"Error calling Nano Banana API: {e}")
            raise e
