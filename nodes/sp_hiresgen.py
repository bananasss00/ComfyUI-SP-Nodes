import comfy, folder_paths, logging
from comfy_execution.graph_utils import GraphBuilder
from collections import namedtuple


class SP_HiresGen_Dynamic:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "start_megapixels": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.5, "max": 3.0, "step": 0.1},
                ),
                "end_megapixels": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.5, "max": 5.0, "step": 0.1},
                ),
                "start_steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "end_steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "start_cfg": (
                    "FLOAT",
                    {"default": 5.5, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "end_cfg": (
                    "FLOAT",
                    {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "start_denoise": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.01, "max": 1.0, "step": 0.05},
                ),
                "end_denoise": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.01, "max": 1.0, "step": 0.05},
                ),
                "start_noise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "end_noise": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("SP_HiresGen_Dynamic",)
    RETURN_NAMES = ("dynamic_cfg",)
    FUNCTION = "doit"

    CATEGORY = "SP-Nodes/HiresGen"

    def doit(self, **kwargs):
        obj = namedtuple("SP_HiresGen_Dynamic", kwargs.keys())
        return (obj(**kwargs),)


class SP_HiresGen_Sharpen:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "sharpen_radius": (
                    "INT",
                    {"default": 1, "min": 1, "max": 31, "step": 1},
                ),
                "sigma": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.1, "max": 10.0, "step": 0.01},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SP_HiresGen_Sharpen",)
    RETURN_NAMES = ("sharpen_cfg",)
    FUNCTION = "doit"

    CATEGORY = "SP-Nodes/HiresGen"

    def doit(self, **kwargs):
        obj = namedtuple("SP_HiresGen_Sharpen", kwargs.keys())
        return (obj(**kwargs),)


class SP_HiresGen_HiresCfg:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list("upscale_models"),
                    {"default": "4x-ultrasharp.pth"},
                ),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 10000}),
                # "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("SP_HiresGen_HiresCfg",)
    RETURN_NAMES = ("hires_cfg",)
    FUNCTION = "doit"

    CATEGORY = "SP-Nodes/HiresGen"

    def doit(self, **kwargs):
        obj = namedtuple("SP_HiresGen_HiresCfg", kwargs.keys())
        return (obj(**kwargs),)


class SP_HiresGen:
    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "hires_cfg": ("SP_HiresGen_HiresCfg",),
                "dynamic_cfg": ("SP_HiresGen_Dynamic",),
                "skip_generation": ("BOOLEAN", {"default": False}),
                "generation_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "dpmpp_2m"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        # https://github.com/Clybius/ComfyUI-Extra-Samplers
                        "default": (
                            "ays"
                            if "ays" in comfy.samplers.KSampler.SCHEDULERS
                            else "karras"
                        )
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 2048, "step": 64},
                ),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "previews": ("BOOLEAN", {"default": False}),
                "color_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sharpen_cfg_opt": ("SP_HiresGen_Sharpen",),
            },
        }
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"
    # OUTPUT_NODE = True

    CATEGORY = "SP-Nodes/HiresGen"

    def doit(
        self,
        model,
        positive,
        negative,
        vae,
        samples,
        hires_cfg,
        dynamic_cfg,
        skip_generation,
        generation_steps,
        sampler_name,
        scheduler,
        seed,
        tile_size,
        tiled_vae,
        previews,
        color_match,
        sharpen_cfg_opt=None,
    ):
        graph = GraphBuilder()
        self.graph = graph
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.samples = samples
        self.hires_cfg = hires_cfg
        self.dynamic_cfg = dynamic_cfg
        self.skip_generation = skip_generation
        self.generation_steps = generation_steps
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.seed = seed
        self.tile_size = tile_size
        self.tiled_vae = tiled_vae
        self.previews = previews
        self.color_match = color_match
        self.sharpen_cfg_opt = sharpen_cfg_opt

        latent = (
            samples
            if skip_generation
            else self.ksampler(samples, generation_steps, dynamic_cfg.start_cfg, 1.0)
        )

        image = None
        image_ref = None

        if self.previews:
            image = self.vae_decode(latent)
            self.preview_image(image)

        for i in range(hires_cfg.iterations):
            current_steps = int(
                round(
                    self.interpolate_step(
                        dynamic_cfg.start_steps,
                        dynamic_cfg.end_steps,
                        i,
                        hires_cfg.iterations,
                    )
                )
            )
            current_cfg = self.interpolate_step(
                dynamic_cfg.start_cfg, dynamic_cfg.end_cfg, i, hires_cfg.iterations
            )
            current_denoise = self.interpolate_step(
                dynamic_cfg.start_denoise,
                dynamic_cfg.end_denoise,
                i,
                hires_cfg.iterations,
            )
            current_noise_strength = self.interpolate_step(
                dynamic_cfg.start_noise, dynamic_cfg.end_noise, i, hires_cfg.iterations
            )
            current_megapixels = self.interpolate_step(
                dynamic_cfg.start_megapixels,
                dynamic_cfg.end_megapixels,
                i,
                hires_cfg.iterations,
            )
            logging.info(
                f"steps: {current_steps}, cfg: {current_cfg}, denoise: {current_denoise}, noise: {current_noise_strength}, megapixels: {current_megapixels}"
            )

            if not image:
                image = self.vae_decode(latent)
                
            if not image_ref:
                image_ref = image

            image = self.upscale_image_with_model(
                image, self.hires_cfg.model, current_megapixels
            )

            if sharpen_cfg_opt:
                image = self.sharpen_image(image)

            latent = self.vae_encode(image)
            latent = self.inject_noise(latent, current_noise_strength)
            latent = self.ksampler(latent, current_steps, current_cfg, current_denoise)

            if i != hires_cfg.iterations - 1:
                if self.previews:
                    image = self.vae_decode(latent)
                    if color_match:
                        image = self.image_color_match(image, image_ref)
                    self.preview_image(image)
            else:
                image = self.vae_decode(latent)
                if color_match:
                    image = self.image_color_match(image, image_ref)

        return {
            "result": (image,),
            "expand": graph.finalize(),
        }

    def interpolate_step(self, start, end, step, total_steps):
        if total_steps == 1:
            return end

        return round(start + (end - start) * (step / (total_steps - 1)), 2)

    def ksampler(self, latent, steps, cfg, denoise):
        ksampler = self.graph.node(
            "KSampler",
            model=self.model,
            positive=self.positive,
            negative=self.negative,
            latent_image=latent,
            seed=self.seed,
            steps=steps,
            cfg=cfg,
            sampler_name=self.sampler_name,
            scheduler=self.scheduler,
            denoise=denoise,
        )
        latent = ksampler.out(0)
        return latent

    def vae_decode(self, latent):
        return self.graph.node(
            "VAEDecodeTiled" if self.tiled_vae else "VAEDecode",
            samples=latent,
            vae=self.vae,
            tile_size=self.tile_size,
        ).out(0)

    def vae_encode(self, image):
        return self.graph.node(
            "VAEEncodeTiled" if self.tiled_vae else "VAEEncode",
            pixels=image,
            vae=self.vae,
            tile_size=self.tile_size,
        ).out(0)

    def sharpen_image(self, image):
        imagesharpen = self.graph.node(
            "ImageSharpen",
            image=image,
            sharpen_radius=self.sharpen_cfg_opt.sharpen_radius,
            sigma=self.sharpen_cfg_opt.sigma,
            alpha=self.sharpen_cfg_opt.alpha,
        )
        return imagesharpen.out(0)

    def inject_noise(self, latent, noise_strength):
        try:
            inject_latent_noise = self.graph.node(
                "InjectLatentNoise+",
                latent=latent,
                noise_seed=self.seed,
                noise_strength=noise_strength,
            )
            latent = inject_latent_noise.out(0)
            return latent
        
        except KeyError:
            raise Exception(f'ComfyUI_essentials required! https://github.com/cubiq/ComfyUI_essentials')

    def image_color_match(self, image, ref):
        try:
            color_space = r"RGB"
            factor = 1
            device = r"auto"
            batch_size = 0

            _image_color_match = self.graph.node('ImageColorMatch+', image=image, reference=ref, color_space=color_space, factor=factor, device=device, batch_size=batch_size)
            image = _image_color_match.out(0)

            return image
        
        except KeyError:
            raise Exception(f'ComfyUI_essentials required! https://github.com/cubiq/ComfyUI_essentials')

    def upscale_image_with_model(self, image, upscale_model, mpx):
        load_upscale_model = self.graph.node(
            "UpscaleModelLoader", model_name=upscale_model
        )
        upscale_model = load_upscale_model.out(0)

        try:
            scale_to_megapixels = self.graph.node(
                "ImageScaleToMegapixels",
                images=image,
                upscale_model_opt=upscale_model,
                megapixels=mpx,
            )
            image = scale_to_megapixels.out(0)
            return image
        
        except KeyError:
            raise Exception(f'comfyui-art-venture required! https://github.com/sipherxyz/comfyui-art-venture')

    def preview_image(self, image):
        self.graph.node("PreviewImage", images=image)


NODE_CLASS_MAPPINGS = {
    "SP_HiresGen_Dynamic": SP_HiresGen_Dynamic,
    "SP_HiresGen_HiresCfg": SP_HiresGen_HiresCfg,
    "SP_HiresGen_Sharpen": SP_HiresGen_Sharpen,
    "SP_HiresGen": SP_HiresGen,
}

"""
PyExec code:

import logging


model=a1
pos=a2
neg=a3
latent=a4
vae=a5
version=a6
previews=a7
seed=a8
tiled=a9

steps = 20
sharpen_radius = 1
sigma = 0.2
alpha = 0.2

tile_size=1024

# artventure
start_megapixels = 0.5
end_megapixels = 2.0
current_megapixels = 0

upscale_type=5 #1-lat_nn, 2-lat_comfy, 3-im_model, 4-im_comfy, 5 - im_model_artventure
upscale_method = r"nearest-exact" # nearest-exact lanczos
upscale_steps = 4
iterations = 3
upscale = 1.25
start_cfg = 5.5
target_cfg = 3.5
start_denoise = 0.55
target_denoise = 0.3
noise_strength = 0.5
target_noise_strength = 0.2

sampler_name = r"dpmpp_2m"
scheduler = r"ays"

def vae_decode(latent):
    return graph.node('VAEDecodeTiled' if tiled else 'VAEDecode', samples=latent, vae=vae, tile_size=tile_size).out(0)
def vae_encode(image):
    return graph.node('VAEEncodeTiled' if tiled else 'VAEEncode', pixels=image, vae=vae, tile_size=tile_size).out(0)

def preview_image(image):
    if not previews:
        return
    graph.node('PreviewImage', images=image)

def sharpen_latent(latent, sharpen_radius, sigma, alpha):
    # LatentOperationSharpen
    latentoperationsharpen = graph.node('LatentOperationSharpen', sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha)
    latent_operation = latentoperationsharpen.out(0)

    # LatentApplyOperation
    latentapplyoperation = graph.node('LatentApplyOperation', samples=latent, operation=latent_operation)
    return latentapplyoperation.out(0)

def sharpen_image(image, sharpen_radius, sigma, alpha):
    # ImageSharpen
    imagesharpen = graph.node('ImageSharpen', image=image, sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha)
    return imagesharpen.out(0)

def upscale_latent(latent, type):
    match type:
        case 1:
            latent = sharpen_latent(latent, sharpen_radius, sigma, alpha)

            # NNLatentUpscale
            nnlatentupscale = graph.node('NNLatentUpscale', latent=latent, version=version, upscale=scale)
            return nnlatentupscale.out(0)
        case 2:
            latent = sharpen_latent(latent, sharpen_radius, sigma, alpha)

            # Upscale Latent By
            upscale_latent_by = graph.node('LatentUpscaleBy', samples=latent, upscale_method=upscale_method, scale_by=upscale)
            return upscale_latent_by.out(0)
        case 3:
            upscale_model = r"4x-ultrasharp.pth"
            mode = r"rescale"
            resize_width = 1024
            resampling_method = r"lanczos"
            supersample = r"true"
            rounding_modulus = 8

            image = vae_decode(latent)
            image = sharpen_image(image, sharpen_radius, sigma, alpha)

            # 🔍 CR Upscale Image
            cr_upscale_image = graph.node('CR Upscale Image', image=image, upscale_model=upscale_model, mode=mode, rescale_factor=upscale, resize_width=resize_width, resampling_method=resampling_method, supersample=supersample, rounding_modulus=rounding_modulus)
            return vae_encode(cr_upscale_image.out(0))
        case 4:
            image = vae_decode(latent)    
            image = sharpen_image(image, sharpen_radius, sigma, alpha)
            # Upscale Image By
            upscale_image_by = graph.node('ImageScaleBy', image=image, upscale_method=upscale_method, scale_by=upscale)
            image_1 = upscale_image_by.out(0)

            return vae_encode(image_1)
        case 5:
            model_name = r"4x-ultrasharp.pth"

            image = vae_decode(latent)
            image = sharpen_image(image, sharpen_radius, sigma, alpha)

            # Load Upscale Model
            load_upscale_model = graph.node('UpscaleModelLoader', model_name=model_name)
            upscale_model = load_upscale_model.out(0)

            # Scale To Megapixels
            scale_to_megapixels = graph.node('ImageScaleToMegapixels', images=image, upscale_model_opt=upscale_model, megapixels=current_megapixels)
            image = scale_to_megapixels.out(0)
            return vae_encode(image)


# KSampler
ksampler = graph.node('KSampler', model=model, positive=pos, negative=neg, latent_image=latent, seed=seed, steps=steps, cfg=start_cfg, sampler_name=sampler_name, scheduler=scheduler, denoise=1.0)
latent = ksampler.out(0)

image = vae_decode(latent)
preview_image(image)

def hires(latent, scale, cfg, denoise, noise_strength):
    latent = upscale_latent(latent, type=upscale_type)

    # 🔧 Inject Latent Noise
    inject_latent_noise = graph.node('InjectLatentNoise+', latent=latent, noise_seed=seed, noise_strength=noise_strength)
    latent = inject_latent_noise.out(0)

    # KSampler
    ksampler_1 = graph.node('KSampler', model=model, positive=pos, negative=neg, latent_image=latent, seed=seed, steps=upscale_steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, denoise=denoise)
    latent = ksampler_1.out(0)

    return latent

def interpolate(start, end, step, total_steps):
    return round(start + (end - start) * (step / (total_steps - 1)), 2)

image = None
for i in range(iterations):
    current_cfg = interpolate(start_cfg, target_cfg, i, iterations)
    current_denoise = interpolate(start_denoise, target_denoise, i, iterations)
    current_noise_strength = interpolate(noise_strength, target_noise_strength, i, iterations)
    current_megapixels = interpolate(start_megapixels, end_megapixels, i, iterations)
    logging.info(f'cfg: {current_cfg}, denoise: {current_denoise}, noise: {current_noise_strength}, megapixels: {current_megapixels}')
    latent = hires(latent, scale=upscale, cfg=current_cfg, denoise=current_denoise, noise_strength=current_noise_strength)

    if i != iterations -1:
        latent = sharpen_latent(latent, sharpen_radius, sigma, alpha)
    image = vae_decode(latent)
    preview_image(image)

r1=image

"""
