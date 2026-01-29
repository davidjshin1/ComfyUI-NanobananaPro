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
    Designed to be a drop-in replacement for GeminiImage2Node but with custom API Key support.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Enter your Gemini API Key"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image you want to generate..."}),
                "images": ("IMAGE",),  # Accepts a batch of images (tensor)
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"], {"default": "3:4"}),
                "resolution": (["1K", "2K", "4K"], {"default": "4K"}),
                "response_modalities": (["IMAGE", "TEXT", "IMAGE,TEXT"], {"default": "IMAGE"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana 🍌"

    # Add version to ensure refresh
    VERSION = "1.1.0"

    def tensor_to_base64(self, image_tensor):
        # Convert single tensor image to base64
        # Input tensor shape: [H, W, C]
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_image(self, api_key, prompt, images, seed, aspect_ratio, resolution, response_modalities):
        if not api_key:
            raise ValueError("API Key is required for Nano Banana Pro.")

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
                "responseModalities": ["TEXT", "IMAGE"], # Always request both to catch text errors/thoughts
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution
                }
            }
        }

        # Add seed if it's not 0 (optional based on API support, but good practice)
        # Note: Gemini API seed support varies, but we pass it if needed or handle logic here.
        # Currently, Gemini 3 API might not strictly respect an integer seed in the payload
        # the same way Stable Diffusion does, but we keep the input for compatibility.

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
                # If safety filters blocked it, the candidate list might be empty or contain finishReason
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
                # Return blank image if failed
                blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (blank, final_text)

            # Concatenate images if multiple returned
            final_image_batch = torch.cat(generated_images, dim=0)
            
            return (final_image_batch, final_text)

        except Exception as e:
            print(f"Error calling Nano Banana API: {e}")
            # Raise to show in ComfyUI
            raise RuntimeError(f"API Error: {e}")
