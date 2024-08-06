import base64
import io
import json
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import struct
import comfy.utils
import time

from server import PromptServer, BinaryEventTypes
from comfy.cli_args import args
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class SP_WebsocketSendImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images, prompt=None, extra_pnginfo=None):
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
   
            byte_io = io.BytesIO()
            img.save(byte_io, format='PNG', pnginfo=metadata)
            
            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.PREVIEW_IMAGE,
                byte_io.getvalue(),
                server.client_id,
            )
            
            results.append(
                # Could put some kind of ID here, but for now just match them by index
                {"source": "websocket", "content-type": "image/png", "type": "output"}
            )
            
        return {"ui": {"images": results}}

    def IS_CHANGED(s, images):
        return time.time()

NODE_CLASS_MAPPINGS = {
    "SP_WebsocketSendImage": SP_WebsocketSendImage,
}
