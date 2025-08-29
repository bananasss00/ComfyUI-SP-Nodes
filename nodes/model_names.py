import folder_paths

def safe_get_filename_list(k):
    try:
        return folder_paths.get_filename_list(k)
    except KeyError:
        return []
    
class SP_Name:
    FOLDER_NAME = "checkpoints"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (safe_get_filename_list(s.FOLDER_NAME), )}}
    
    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_name"
    CATEGORY = "SP-Nodes/ModelName"

    def get_name(self, name):
        return name,

class SP_Name_Checkpoint(SP_Name):
    FOLDER_NAME = "checkpoints"
class SP_Name_Unet(SP_Name):
    FOLDER_NAME = "diffusion_models"
class SP_Name_ControlNet(SP_Name):
    FOLDER_NAME = "controlnet"
class SP_Name_UpscaleModel(SP_Name):
    FOLDER_NAME = "upscale_models"
class SP_Name_StyleModel(SP_Name):
    FOLDER_NAME = "style_models"
class SP_Name_ClipVision(SP_Name):
    FOLDER_NAME = "clip_vision"

class SP_Name_Clip(SP_Name):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (s.get_clip_list(), )}}

    @classmethod
    def get_clip_list(s):
        files = []
        files += safe_get_filename_list("text_encoders")
        files += safe_get_filename_list("clip_gguf")
        return sorted(files)

class SP_Name_Vae(SP_Name):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (s.vae_list(), )}}

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes



NODE_CLASS_MAPPINGS = {
    "SP_Name_Checkpoint": SP_Name_Checkpoint,
    "SP_Name_Unet": SP_Name_Unet,
    "SP_Name_Clip": SP_Name_Clip,
    "SP_Name_Vae": SP_Name_Vae,
    "SP_Name_ControlNet": SP_Name_ControlNet,
    "SP_Name_UpscaleModel": SP_Name_UpscaleModel,
    "SP_Name_StyleModel": SP_Name_StyleModel,
    "SP_Name_ClipVision": SP_Name_ClipVision,
}