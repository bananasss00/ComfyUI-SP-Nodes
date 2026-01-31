

import json
import os
import server
import torch
import hashlib
import comfy.utils
import safetensors.torch

class SP_SaveLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ),
                              "filename": ("STRING", {"default": "abs_filepath.latent"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save"

    OUTPUT_NODE = True

    CATEGORY = 'SP-Nodes/latent'

    def save(self, samples, filename="abs_filepath.latent", prompt=None, extra_pnginfo=None):
        # Проверяем, абсолютный ли путь
        if not os.path.isabs(filename):
            filename = os.path.abspath(filename)
        # Получаем директорию и имя файла
        full_output_folder = os.path.dirname(filename)
        file = os.path.basename(filename)

        # Создаем директорию, если она не существует
        if full_output_folder and not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        # support save metadata for latent sharing
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = None
        metadata = {"prompt": prompt_info}
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

        results: list = []
        results.append({
            "filename": file,
            "subfolder": full_output_folder,
            "type": "output"
        })

        file_path = os.path.join(full_output_folder, file)

        output = {}
        output["latent_tensor"] = samples["samples"].contiguous()
        output["latent_format_version_0"] = torch.tensor([])

        comfy.utils.save_torch_file(output, file_path, metadata=metadata)
        return { "ui": { "latents": results } }


class SP_LoadLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filename": ("STRING", {"default": "abs_path_to_latent..."})}, }

    CATEGORY = "SP-Nodes/latent"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "load"

    def load(self, filename):
        if not os.path.isabs(filename):
            latent_path = os.path.abspath(filename)
        else:
            latent_path = filename
            
        # Загружаем файл напрямую по абсолютному пути
        latent = safetensors.torch.load_file(latent_path, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
        samples = {"samples": latent["latent_tensor"].float() * multiplier}
        return (samples, )

    @classmethod
    def IS_CHANGED(s, filename):
        if not os.path.isabs(filename):
            image_path = os.path.abspath(filename)
        else:
            image_path = filename
        m = hashlib.sha256()
        try:
            with open(image_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        except Exception:
            return None

    @classmethod
    def VALIDATE_INPUTS(s, filename):
        if not os.path.isabs(filename):
            image_path = os.path.abspath(filename)
        else:
            image_path = filename
        if not os.path.exists(image_path):
            return f"Invalid latent file: {image_path}"
        return True
    

NODE_CLASS_MAPPINGS = {
    "SP_SaveLatent": SP_SaveLatent,
    "SP_LoadLatent": SP_LoadLatent,
}
