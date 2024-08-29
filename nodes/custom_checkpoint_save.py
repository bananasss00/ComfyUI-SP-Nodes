import comfy.sd
import comfy.utils
import comfy.model_base
import comfy.model_sampling

import torch
import folder_paths
import json
import os

from comfy import model_management
from comfy.cli_args import args

# custom comfy.sd.save_checkpoint
def comfy_sd_save_checkpoint(output_path, model, clip=None, vae=None, clip_vision=None, metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()

    model_management.load_models_gpu(load_models)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    vae_sd = vae.get_sd() if vae is not None else None
    sd = model.model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)
      
def save_checkpoint(model, clip=None, vae=None, clip_vision=None, filename_prefix=None, output_dir=None, prompt=None, extra_pnginfo=None):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {}

    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    #TODO:
    # "stable-diffusion-v1", "stable-diffusion-v1-inpainting", "stable-diffusion-v2-512",
    # "stable-diffusion-v2-768-v", "stable-diffusion-v2-unclip-l", "stable-diffusion-v2-unclip-h",
    # "v2-inpainting"

    extra_keys = {}
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"

    if not args.disable_metadata:
        metadata["prompt"] = prompt_info
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    output_checkpoint = f"{filename}_{counter:05}_.safetensors"
    output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

    comfy_sd_save_checkpoint(output_checkpoint, model, clip, vae, clip_vision, metadata=metadata, extra_keys=extra_keys)
    
class CheckpointSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),},
                "optional": { "clip_opt": ("CLIP",),
                              "vae_opt": ("VAE",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "SP-Nodes"

    def save(self, model, filename_prefix, clip_opt=None, vae_opt=None, prompt=None, extra_pnginfo=None):
        save_checkpoint(model, clip=clip_opt, vae=vae_opt, filename_prefix=filename_prefix, output_dir=self.output_dir, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}

class SP_UnetSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "SP-Nodes"

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        feature_prefix="model.diffusion_model."
        state_dict = model.model.state_dict_for_saving()

        # extract unet
        state_dict = {
                k.replace(f"{feature_prefix}", ""): v
                for k, v in state_dict.items()
                if k.startswith(feature_prefix)
            }

        for k in state_dict:
                t = state_dict[k]
                if not t.is_contiguous():
                    state_dict[k] = t.contiguous()

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        # metadata
        metadata = {}

        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        enable_modelspec = True
        if isinstance(model.model, comfy.model_base.SDXL):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
        elif isinstance(model.model, comfy.model_base.SDXLRefiner):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
        else:
            enable_modelspec = False

        if enable_modelspec:
            metadata["modelspec.sai_model_spec"] = "1.0.0"
            metadata["modelspec.implementation"] = "sgm"
            metadata["modelspec.title"] = "{} {}".format(filename, counter)

        if model.model.model_type == comfy.model_base.ModelType.EPS:
            metadata["modelspec.predict_key"] = "epsilon"
        elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
            metadata["modelspec.predict_key"] = "v"

        if not args.disable_metadata:
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])
        
        # saving
        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(state_dict, output_checkpoint, metadata=metadata)
        return {}

NODE_CLASS_MAPPINGS = {
    "SP-CheckpointSave": CheckpointSave,
    "SP-UnetSave": SP_UnetSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SP-CheckpointSave": "SP Custom Checkpoint Save",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]