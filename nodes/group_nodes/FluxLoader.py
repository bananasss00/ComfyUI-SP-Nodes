import folder_paths, nodes, logging, comfy
from comfy_execution.graph import DynamicPrompt

from .Graph import Graph

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

MAX_RESOLUTION = 16384

BASE_RESOLUTIONS = [
    ("width", "height"),
    (512, 512),
    (512, 768),
    (576, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (832, 1216),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1080, 1920),
    (1440, 2560),
    (1088, 896),
    (1216, 832),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    (1920, 816),
    (1920, 1080),
    (2560, 1440),
]

resolution_strings = [
    (
        f"{width} x {height} (custom)"
        if width == "width" and height == "height"
        else f"{width} x {height}"
    )
    for width, height in BASE_RESOLUTIONS
]

CONTEXT = {
    "sp_pipe": "SP_PIPE",
    "model": "MODEL",
    "clip": "CLIP",
    "vae": "VAE",
    "positive": "CONDITIONING",
    "negative": "CONDITIONING",
    "latent": "LATENT",
    "image": "IMAGE",
}

def load_unet_and_dual_clip(
        unet_name,
        weight_dtype,
        vae_name,
        clip_name1,
        clip_name2,
        lora_name,
        lora_strength,
        positive,
        resolution,
        empty_latent_width,
        empty_latent_height,
        clip_type='flux',
        clip_name3=None,
        negative=None,
        de_distilled=False
    ):
    graph = Graph()

    # set resolution
    if resolution not in ["width x height (custom)"]:
        try:
            width, height = map(int, resolution.split(" x "))
            empty_latent_width = width
            empty_latent_height = height
        except ValueError:
            raise ValueError("Invalid base_resolution format.")

    # load unet
    model = None
    if weight_dtype.startswith("fp8") or weight_dtype == "default":
        model = graph.UNETLoader(unet_name, weight_dtype)
    elif weight_dtype.startswith("nf4"):
        model = graph.SP_UnetLoaderBNB(unet_name, weight_dtype.split("-")[1])
    elif weight_dtype == "gguf":
        model = graph.UnetLoaderGGUF(unet_name)

    clip = None
    if clip_name2 == "None":
        clip = graph.CLIPLoaderGGUF(clip_name1)
    elif clip_name3 == "None":
        clip = graph.DualCLIPLoaderGGUF(clip_name1, clip_name2, clip_type)
    else:
        clip = graph.TripleCLIPLoaderGGUF(clip_name1, clip_name2, clip_name3)

    vae = graph.VAELoader(vae_name)

    if lora_name != "None":
        model, clip = graph.LoraLoader(model, clip, lora_name, lora_strength)

    pos = neg = graph.CLIPTextEncode(clip, positive)
    if negative is not None and clip_type == "sd3":
        neg = graph.CLIPTextEncode(clip, negative)
        zero_out = graph.ConditioningZeroOut(neg)
        cond1 = graph.ConditioningSetTimestepRange(neg, start=0.0, end=0.1)
        cond2 = graph.ConditioningSetTimestepRange(zero_out, start=0.1, end=1.0)
        neg = graph.ConditioningCombine(cond1, cond2)
    elif negative is not None:
        neg = graph.CLIPTextEncode(clip, negative)

    # model = graph.ModelSamplingFlux(model, max_shift, base_shift, empty_latent_width, empty_latent_height)

    latent = graph.EmptySD3LatentImage(empty_latent_width, empty_latent_height, 1)

    # SP_Pipe
    sp_pipe = graph.SP_Pipe(
        None,
        model,
        clip,
        vae,
        pos,
        neg,
        latent,
        None,
        _model_type=("flux" if clip_type == "flux" and not de_distilled else None),
    )[0]

    return {
        "result": (sp_pipe, model, clip, vae, pos, neg, latent),
        "expand": graph.finalize(),
    }

def ksampler_main(
    sp_pipe,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    denoise,
    inject_noise,
    tile_size,
    vae_decode,
    preview,
    model=None,
    latent_image=None,
    image=None,
):
    graph = Graph()

    # get values from linked pipe. all values also LINKS!
    sp_pipe, pipe_model, clip, vae, positive, negative, latent, pipe_image = (
        graph.SP_Pipe(sp_pipe)
    )

    model = graph.AnySwitch(model, pipe_model)

    def get_prior_latent(image, pipe_image, latent_image, latent):
        # workflow: .assets\select_latent_or_encoded_latent_ref.png

        # Selects the image that is not None, either from the first or second input.
        image = graph.AnySwitch(image, pipe_image)

        # Checks if 'image' is None or not and returns a boolean flag.
        # 'image_not_none' will be True if 'image' is valid (not None).
        _, image_not_none = graph.ImpactIfNone(any_input=image)

        # Encodes the selected image into a latent representation using the VAE.
        # If 'tile_size' is specified, the encoding will be tiled accordingly.
        encoded_latent = graph.VAEEncode(image, vae, tile_size)

        # Selects the latent image that is not None from two possible inputs.
        latent = graph.AnySwitch(latent_image, latent)

        # If the 'image' was valid (not None), use 'encoded_latent'.
        # Otherwise, use the pre-generated latent from the pipeline or a custom one provided to the node.
        latent = graph.ImpactConditionalBranch(
            tt_value=encoded_latent,  # Use this if 'image_not_none' is True.
            ff_value=latent,  # Use this if 'image_not_none' is False.
            cond=image_not_none,  # Condition based on whether 'image' was valid.
        )

        return latent

    latent = get_prior_latent(image, pipe_image, latent_image, latent)

    if inject_noise > 0.01:
        latent = graph.InjectLatentNoise(latent, noise_seed=seed, noise_strength=inject_noise)

    def flux_patches(positive, cfg):
        model_type = graph.SP_DictValue(sp_pipe, '_model_type')
        is_flux = graph.ImpactCompare('flux', model_type)
        positive = graph.ImpactConditionalBranch(tt_value=graph.FluxGuidance(positive, cfg), ff_value=positive, cond=is_flux)
        cfg = graph.ImpactConditionalBranch(tt_value=1.0, ff_value=cfg, cond=is_flux)
        return positive, cfg
    
    positive, cfg = flux_patches(positive, cfg)

    latent = graph.KSampler(
        model,
        positive,
        negative,
        latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
    )

    image = None
    if vae_decode or preview:
        image = graph.VAEDecode(latent, vae, tile_size)

        if preview:
            graph.PreviewImage(image)
    
    sp_pipe = graph.SP_Pipe(
        sp_pipe,
        model,
        clip,
        vae,
        positive,
        negative,
        latent,
        None
    )[0]
    
    return {
        "result": (sp_pipe, latent, image),
        "expand": graph.finalize(),
    }

class SP_DictValue:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dictionary": (AnyType("*"),),
                "key": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = AnyType('*'),
    FUNCTION = "fn"

    def fn(self, dictionary: dict, key):
        return dictionary.get(key, None),

class SP_Pipe:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {k: tuple([v]) for k, v in CONTEXT.items()},
            "hidden": {'_model_type': "STRING"}
        }

    RETURN_TYPES = tuple(CONTEXT.values())
    RETURN_NAMES = tuple(CONTEXT.keys())
    FUNCTION = "fn"

    def fn(self, sp_pipe=None, **kwargs):
        if not sp_pipe:
            sp_pipe = {k: None for k in CONTEXT.keys() if k != "sp_pipe"}
        else:
            sp_pipe = dict(sp_pipe)

        for k, v in kwargs.items():
            if v is None or k == "sp_pipe":
                continue
            sp_pipe[k] = v

        result = tuple([sp_pipe] + [v for k, v in sp_pipe.items() if not k.startswith("_")])

        return result

class SP_DDInpaint_Pipe:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sp_pipe": ("SP_PIPE",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = "SP_PIPE",
    RETURN_NAMES = "sp_pipe",
    FUNCTION = "fn"

    def fn(self, sp_pipe, pixels, mask):
        graph = Graph()

        model = graph.DifferentialDiffusion(sp_pipe['model'])
        positive, negative, latent = graph.InpaintModelConditioning(sp_pipe['positive'], sp_pipe['negative'], sp_pipe['vae'], pixels, mask)

        sp_pipe = graph.SP_Pipe(sp_pipe=sp_pipe, model=model, positive=positive, negative=negative, latent=latent)[0]

        return {
            "result": (sp_pipe, ),
            "expand": graph.finalize(),
        }


class SP_SDLoader:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked"] + nodes.VAELoader.vae_list(),),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05},
                ),
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "placeholder": "positive",
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "placeholder": "negative",
                    },
                ),
                "stop_at_clip_layer": (
                    "INT",
                    {"default": -1, "min": -24, "max": -1, "step": 1},
                ),
                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "empty_latent_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "empty_latent_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
            },
            "optional": {},
        }
        return inputs

    RETURN_TYPES = (
        "SP_PIPE",
        "MODEL",
        "CLIP",
        "VAE",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
    )
    RETURN_NAMES = ("sp_pipe", "model", "clip", "vae", "positive", "negative", "latent")
    FUNCTION = "fn"

    def fn(
        self,
        ckpt_name,
        vae_name,
        lora_name,
        lora_strength,
        positive,
        negative,
        stop_at_clip_layer,
        resolution,
        empty_latent_width,
        empty_latent_height,
    ):
        graph = Graph()

        # set resolution
        if resolution not in ["width x height (custom)"]:
            try:
                width, height = map(int, resolution.split(" x "))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        model, clip, vae = graph.CheckpointLoaderSimple(ckpt_name)

        if vae_name != "Baked":
            vae = graph.VAELoader(vae_name)

        # Load LoRA
        if lora_name != "None":
            model, clip = graph.LoraLoader(model, clip, lora_name, lora_strength)

        clip = graph.CLIPSetLastLayer(clip, stop_at_clip_layer)

        pos = graph.CLIPTextEncode(clip, positive)
        neg = graph.CLIPTextEncode(clip, negative)

        latent = graph.EmptyLatentImage(empty_latent_width, empty_latent_height, 1)

        sp_pipe = graph.SP_Pipe(None, model, clip, vae, pos, neg, latent, None)[0]

        return {
            "result": (sp_pipe, model, clip, vae, pos, neg, latent),
            "expand": graph.finalize(),
        }


class SP_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sp_pipe": (
                    "SP_PIPE",
                    {"rawLink": True},
                ),  # use link for better caching inside node
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "beta",
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
                "inject_noise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.05,
                    },
                ),
                "tile_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 64},
                ),
                "vae_decode": ("BOOLEAN", {"default": True}),
                "preview": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to.",
                        "rawLink": True,
                    },
                ),
                "latent_image": (
                    "LATENT",
                    {"tooltip": "The latent image to denoise.", "rawLink": True},
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "The image to denoise.", "rawLink": True},
                ),
            },
        }

    RETURN_TYPES = ("SP_PIPE", "LATENT", "IMAGE")
    OUTPUT_TOOLTIPS = ("The denoised latent.", "The decoded image")
    FUNCTION = "fn"
    OUTPUT_NODE = True

    CATEGORY = "SP-Nodes/Group Nodes"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def fn(
        self,
        sp_pipe,
        seed,
        steps,
        sampler_name,
        scheduler,
        denoise,
        inject_noise,
        tile_size,
        vae_decode,
        preview,
        cfg=1.0,
        model=None,
        latent_image=None,
        image=None,
    ):
        return ksampler_main(
            sp_pipe,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            inject_noise,
            tile_size,
            vae_decode,
            preview,
            model,
            latent_image,
            image,
        )


class SP_FluxLoader:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "unet_name": (
                    [
                        x
                        for x in folder_paths.get_filename_list("diffusion_models")
                        + folder_paths.get_filename_list("unet_gguf")
                    ],
                ),
                "de_distilled": ("BOOLEAN", {"default": False}),
                "weight_dtype": (
                    [
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                        "nf4-float8_e4m3fn",
                        "nf4-float8_e5m2",
                        "gguf",
                    ],
                ),
                "vae_name": (nodes.VAELoader.vae_list(),),
                "clip_name1": (self.get_clip_list(),),
                "clip_name2": (self.get_clip_list(),),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05},
                ),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "positive"}),
                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "empty_latent_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "empty_latent_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
            },
            "optional": {},
        }
        return inputs

    RETURN_TYPES = ("SP_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("sp_pipe", "model", "clip", "vae", "positive", "negative", "latent")
    FUNCTION = "fn"

    def fn(
        self,
        unet_name,
        de_distilled,
        weight_dtype,
        vae_name,
        clip_name1,
        clip_name2,
        lora_name,
        lora_strength,
        positive,
        resolution,
        empty_latent_width,
        empty_latent_height,
        clip_type = 'flux', clip_name3 = "None", negative = None
    ):
        return load_unet_and_dual_clip(unet_name,
            weight_dtype,
            vae_name,
            clip_name1,
            clip_name2,
            lora_name,
            lora_strength,
            positive,
            resolution,
            empty_latent_width,
            empty_latent_height, clip_type, clip_name3, negative, de_distilled)

    @classmethod
    def get_clip_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

class SP_SD3Loader(SP_FluxLoader):
    @classmethod
    def INPUT_TYPES(self):
        input_types = dict(super().INPUT_TYPES())

        required = input_types["required"]
        
        new_required = {}
        for key, value in required.items():
            new_required[key] = value

            if key == "clip_name2":
                new_required["clip_name2"] = new_required["clip_name3"] = (["None"] + self.get_clip_list(),)
            elif key == "positive":
                new_required["negative"] = ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "negative"})                
        
        del new_required['de_distilled']

        new_required['weight_dtype'] = (
                    [
                        "fp8_e4m3fn",
                        "fp8_e5m2",
                        "nf4-float8_e4m3fn",
                        "nf4-float8_e5m2",
                        "gguf",
                        "default"
                    ],)

        input_types["required"] = new_required

        return input_types

    def fn(
        self,
        unet_name,
        weight_dtype,
        vae_name,
        clip_name1,
        clip_name2,
        lora_name,
        lora_strength,
        positive,
        resolution,
        empty_latent_width,
        empty_latent_height,
        clip_type = 'sd3', clip_name3 = "None", negative = None
    ):
        return super().fn(unet_name,
            False,
            weight_dtype,
            vae_name,
            clip_name1,
            clip_name2,
            lora_name,
            lora_strength,
            positive,
            resolution,
            empty_latent_width,
            empty_latent_height, clip_type, clip_name3, negative)

NODE_CLASS_MAPPINGS = {
    "SP_Pipe": SP_Pipe,
    "SP_SDLoader": SP_SDLoader,
    "SP_KSampler": SP_KSampler,
    "SP_FluxLoader": SP_FluxLoader,
    "SP_SD3Loader": SP_SD3Loader,
    "SP_DictValue": SP_DictValue,
    "SP_DDInpaint_Pipe": SP_DDInpaint_Pipe,
}
