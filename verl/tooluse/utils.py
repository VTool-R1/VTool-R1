import json
from PIL import Image

def custom_encoder(obj):
    """Custom JSON encoder function that replaces Image objects with '<image>'.
       Delegates the encoding of other types to the default encoder."""
    if isinstance(obj, Image.Image):
        return "<image>"
    # Let the default JSON encoder handle any other types
    return json.JSONEncoder().default(obj)