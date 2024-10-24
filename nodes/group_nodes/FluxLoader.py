import folder_paths, nodes, logging, comfy
from comfy_execution.graph_utils import GraphBuilder


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


class SP_Pipe:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {k: tuple([v]) for k, v in CONTEXT.items()},
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
            if v == None or k == "sp_pipe":
                continue
            sp_pipe[k] = v

        return tuple([sp_pipe, *sp_pipe.values()])


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
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
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

    RETURN_TYPES = ("SP_PIPE",)
    RETURN_NAMES = ("sp_pipe",)
    FUNCTION = "fn"

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
    ):
        graph = GraphBuilder()

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
        if weight_dtype.startswith("fp8"):
            # Load Diffusion Model
            load_diffusion_model = graph.node(
                "UNETLoader", unet_name=unet_name, weight_dtype=weight_dtype
            )
            model = load_diffusion_model.out(0)
        elif weight_dtype.startswith("nf4"):
            # SP_UnetLoaderBNB
            sp_unetloaderbnb = graph.node(
                "SP_UnetLoaderBNB",
                unet_name=unet_name,
                load_dtype=weight_dtype.split("-")[1],
            )
            model = sp_unetloaderbnb.out(0)
        elif weight_dtype == "gguf":
            # Unet Loader (GGUF)
            unet_loader_gguf = graph.node("UnetLoaderGGUF", unet_name=unet_name)
            model = unet_loader_gguf.out(0)

        # DualCLIPLoader (GGUF)
        dualcliploader_gguf = graph.node(
            "DualCLIPLoaderGGUF",
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            type="flux",
        )
        clip = dualcliploader_gguf.out(0)

        # Load VAE
        load_vae = graph.node("VAELoader", vae_name=vae_name)
        vae = load_vae.out(0)

        # Load LoRA
        if lora_name != "None":
            load_lora = graph.node(
                "LoraLoader",
                model=model,
                clip=clip,
                lora_name=lora_name,
                strength_model=lora_strength,
                strength_clip=lora_strength,
            )
            model = load_lora.out(0)
            clip = load_lora.out(1)

        # CLIP Text Encode (Prompt)
        clip_text_encode_prompt = graph.node("CLIPTextEncode", clip=clip, text=positive)
        conditioning = clip_text_encode_prompt.out(0)

        # EmptySD3LatentImage
        empty_latent_image = graph.node(
            "EmptySD3LatentImage",
            width=empty_latent_width,
            height=empty_latent_height,
            batch_size=1,
        )
        latent = empty_latent_image.out(0)

        # SP_Pipe
        sp_pipe = graph.node(
            "SP_Pipe",
            sp_pipe=None,
            model=model,
            clip=clip,
            vae=vae,
            positive=conditioning,
            negative=None,
            latent=latent,
            image=None,
        )

        return {
            "result": (sp_pipe.out(0),),
            "expand": graph.finalize(),
        }

    @classmethod
    def get_clip_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)


class SP_FluxKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sp_pipe": ("SP_PIPE",),
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
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image."
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 64},
                ),
                "vae_decode": ("BOOLEAN", {"default": True}),
                "preview": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "image": ("IMAGE", {"tooltip": "The image to denoise."}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
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
        tile_size,
        vae_decode,
        preview,
        latent_image=None,
        image=None,
    ):
        graph = GraphBuilder()

        latent = None
        if image is not None:
            if tile_size:
                # VAE Encode (Tiled)
                vae_encode_tiled = graph.node(
                    "VAEEncodeTiled",
                    pixels=image,
                    vae=sp_pipe["vae"],
                    tile_size=tile_size,
                )
                latent = vae_encode_tiled.out(0)
            else:
                # VAE Encode
                vae_encode = graph.node("VAEEncode", pixels=image, vae=sp_pipe["vae"])
                latent = vae_encode.out(0)
        elif latent_image:
            latent = latent_image
        else:
            latent = sp_pipe["latent"]

        # KSampler
        ksampler = graph.node(
            "KSampler",
            model=sp_pipe["model"],
            positive=sp_pipe["positive"],
            negative=sp_pipe["positive"],
            latent_image=latent,
            seed=seed,
            steps=steps,
            cfg=1.0,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )
        latent = ksampler.out(0)

        image = None
        if vae_decode or preview:
            if tile_size:
                # VAE Decode (Tiled)
                vae_decode_tiled = graph.node(
                    "VAEDecodeTiled",
                    samples=latent,
                    vae=sp_pipe["vae"],
                    tile_size=tile_size,
                )
                image = vae_decode_tiled.out(0)
            else:
                # VAE Decode
                vae_decode = graph.node("VAEDecode", samples=latent, vae=sp_pipe["vae"])
                image = vae_decode.out(0)

            if preview:
                # Preview Image
                preview_image = graph.node("PreviewImage", images=image)

        return {
            "result": (latent, image),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "SP_Pipe": SP_Pipe,
    "SP_FluxLoader": SP_FluxLoader,
    "SP_FluxKSampler": SP_FluxKSampler,
}
