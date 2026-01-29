from .nanobanana_node import NanoBananaProNode

NODE_CLASS_MAPPINGS = {
    "NanoBananaPro": NanoBananaProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaPro": "🍌 Nano Banana Pro (Gemini 3)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
