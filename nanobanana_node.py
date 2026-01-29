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
    Drop-in replacement for GeminiImage2Node with direct API support.
    Accepts client connection from GeminiAPIConfig node.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image you want to generate...", "forceInput": True}),
                "images": ("IMAGE",),
                "client": ("GEMINI_CLIENT",),  # Connection from GeminiAPIConfig
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"], {"default": "3:4"}),
                "resolution": (["1K", "2K", "4K"], {"default": "4K"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana 🍌"

    VERSION = "1.2.0"

    def tensor_to_base64(self, image_tensor):
        # Convert single tensor image to base64
        # Input tensor shape: [H, W, C]
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_api_key(self, client):
        """Extract API key from various client object types."""
        # If it's already a string, use it directly
        if isinstance(client, str):
            return client
        
        # If it has an api_key attribute (common pattern)
        if hasattr(client, 'api_key'):
            return client.api_key
        
        # If it's a dict-like object
        if hasattr(client, 'get'):
            return client.get('api_key', None)
        
        # If it's a tuple (sometimes ComfyUI passes tuples)
        if isinstance(client, tuple) and len(client) > 0:
            return self.extract_api_key(client[0])
        
        # Last resort - try to convert to string
        return str(client)

    def generate_image(self, prompt, images, client, seed, aspect_ratio, resolution):
        # Extract API key from client object
        api_key = self.extract_api_key(client)
        
        if not api_key:
            raise ValueError("Could not extract API Key from client connection.")

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        
        # Prepare content parts
        parts = [{"text": prompt}]
        
        # Process input images (handle batch)
        # images shape is [B, H, W, C]
        if images is not None:
            batch_size = images.shape[0]
            # Limit to max 14 images as per Gemini 3 specs
            limit = min(batch_size, 14)
            
            for i in range(limit):
                single_image = images[i]
                b64_data = self.tensor_to_base64(single_image)
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
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            generated_images = []
            response_texts = []

            candidates = result.get('candidates', [])
            if not candidates:
                raise ValueError(f"No candidates returned. Response: {json.dumps(result)}")

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
            
            final_text = "\n".join(response_texts)

            if not generated_images:
                print(f"Warning: No image returned. Text: {final_text}")
                blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (blank, final_text)

            final_image_batch = torch.cat(generated_images, dim=0)
            
            return (final_image_batch, final_text)

        except Exception as e:
            print(f"Error calling Nano Banana API: {e}")
            raise RuntimeError(f"API Error: {e}")
