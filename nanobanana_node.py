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
    Drop-in replacement for GeminiImage2Node with GeminiAPIConfig client support.
    Connects directly to Google's official API.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"], {"default": "gemini-3-pro-image-preview"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_modalities": (["IMAGE", "TEXT", "IMAGE+TEXT"], {"default": "IMAGE+TEXT"}),
                "client": ("GEMINI_CLIENT",),  # Connection from GeminiAPIConfig
            },
            "optional": {
                "images": ("IMAGE",),  # Optional reference images
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana 🍌"

    VERSION = "1.3.0"

    def tensor_to_base64(self, image_tensor):
        # Convert single tensor image to base64
        # Input tensor shape: [H, W, C]
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_api_key(self, client):
        """Extract API key from GeminiAPIConfig client object."""
        # If it's already a string, use it directly
        if isinstance(client, str):
            return client
        
        # If it has an api_key attribute (common pattern)
        if hasattr(client, 'api_key'):
            return client.api_key
        
        # If it's a dict-like object
        if hasattr(client, 'get'):
            key = client.get('api_key', None)
            if key:
                return key
        
        # If it's a tuple (sometimes ComfyUI passes tuples)
        if isinstance(client, tuple) and len(client) > 0:
            return self.extract_api_key(client[0])
        
        # Try __dict__ for object attributes
        if hasattr(client, '__dict__'):
            if 'api_key' in client.__dict__:
                return client.__dict__['api_key']
        
        # Last resort - try to convert to string (might be the key itself)
        result = str(client)
        if result and len(result) > 10 and not result.startswith('<'):
            return result
            
        return None

    def generate_image(self, prompt, model, seed, aspect_ratio, resolution, response_modalities, client, images=None):
        # Extract API key from client object
        api_key = self.extract_api_key(client)
        
        if not api_key:
            raise ValueError(f"Could not extract API Key from client connection. Client type: {type(client)}, value: {repr(client)[:100]}")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # Prepare content parts
        parts = [{"text": prompt}]
        
        # Process input images (handle batch) if provided
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

        # Parse response modalities
        modalities = []
        if "TEXT" in response_modalities:
            modalities.append("TEXT")
        if "IMAGE" in response_modalities:
            modalities.append("IMAGE")
        if not modalities:
            modalities = ["TEXT", "IMAGE"]

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "responseModalities": modalities,
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
            print(f"[NanoBananaPro] Calling {model} with aspect_ratio={aspect_ratio}, resolution={resolution}")
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            generated_images = []
            response_texts = []

            candidates = result.get('candidates', [])
            if not candidates:
                error_info = json.dumps(result, indent=2)[:500]
                raise ValueError(f"No candidates returned. Response: {error_info}")

            for part in candidates[0].get('content', {}).get('parts', []):
                if 'text' in part:
                    response_texts.append(part['text'])
                
                if 'inlineData' in part:
                    img_data = base64.b64decode(part['inlineData']['data'])
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    
                    # Convert PIL image to torch tensor [H, W, C]
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np)[None,]
                    generated_images.append(img_tensor)
            
            final_text = "\n".join(response_texts)

            if not generated_images:
                print(f"[NanoBananaPro] Warning: No image returned. Text: {final_text[:200]}")
                # Return blank image if failed
                blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (blank, final_text)

            # Concatenate images if multiple returned
            final_image_batch = torch.cat(generated_images, dim=0)
            print(f"[NanoBananaPro] Successfully generated {len(generated_images)} image(s)")
            
            return (final_image_batch, final_text)

        except requests.exceptions.RequestException as e:
            print(f"[NanoBananaPro] Network error: {e}")
            raise RuntimeError(f"API Network Error: {e}")
        except Exception as e:
            print(f"[NanoBananaPro] Error: {e}")
            raise RuntimeError(f"API Error: {e}")
