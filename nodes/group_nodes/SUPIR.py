import folder_paths, nodes, logging, comfy
from comfy_execution.graph import DynamicPrompt

from .Graph import Graph


class SP_SupirSampler:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "supir_sampler": ("SP_SupirSampler",),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("dpmpp_eta", "edm_s_churn", "restore_cfg", "sampler")
    FUNCTION = "fn"

    def fn(self, supir_sampler):
        logging.info(f'supir_sampler: {supir_sampler}')
        return supir_sampler


class SP_SupirSampler_DPMPP2M:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "dpmpp_eta": (
                    "FLOAT",
                    {"default": 0.1, "min": 0, "max": 10.0, "step": 0.01},
                ),
                "tiled": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SP_SupirSampler",)
    RETURN_NAMES = ("supir_sampler",)
    FUNCTION = "fn"

    def fn(self, dpmpp_eta, tiled):
        return (
            dpmpp_eta,
            5,
            -1,
            ("TiledRestoreDPMPP2MSampler" if tiled else "RestoreDPMPP2MSampler"),
        ),


class SP_SupirSampler_EDM:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "edm_s_churn": ("INT", {"default": 5, "min": 0, "max": 40, "step": 1}),
                "restore_cfg": (
                    "FLOAT",
                    {"default": 1, "min": -1.0, "max": 20.0, "step": 0.01},
                ),
                "tiled": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SP_SupirSampler",)
    RETURN_NAMES = ("supir_sampler",)
    FUNCTION = "fn"

    def fn(self, edm_s_churn, restore_cfg, tiled):
        return (
            1.0,
            edm_s_churn,
            restore_cfg,
            ("TiledRestoreEDMSampler" if tiled else "RestoreEDMSampler"),
        ),


class SP_Supir:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "model": ("MODEL", {"rawLink": True}),
                "clip": ("CLIP", {"rawLink": True}),
                "vae": ("VAE", {"rawLink": True}),
                "image": ("IMAGE", {"rawLink": True}),
                "supir_sampler": ("SP_SupirSampler", {"rawLink": True}),
                "supir_model": (folder_paths.get_filename_list("checkpoints"),),
                "fp8_unet": ("BOOLEAN", {"default": False}),
                "vae_tile_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 64},
                ),
                "sampler_tile_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 64},
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "high quality, detailed",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "bad quality, blurry, messy",
                    },
                ),
                "steps": ("INT", {"default": 45, "min": 3, "max": 100, "step": 1}),
                "cfg_scale_start": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "cfg_scale_end": (
                    "FLOAT",
                    {"default": 4.0, "min": 0, "max": 100.0, "step": 0.01},
                ),
                "s_noise": (
                    "FLOAT",
                    {"default": 1.003, "min": 1.0, "max": 1.1, "step": 0.001},
                ),
                "control_scale_start": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01},
                ),
                "control_scale_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": 123, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
                "color_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "denoised_image")
    FUNCTION = "fn"

    def fn(
        self,
        model,
        vae,
        clip,
        image,
        supir_sampler,
        supir_model,
        fp8_unet,
        vae_tile_size,
        sampler_tile_size,
        positive_prompt,
        negative_prompt,
        steps,
        cfg_scale_start,
        cfg_scale_end,
        s_noise,
        control_scale_start,
        control_scale_end,
        seed,
        color_match,
    ):
        return self.supir(
            model,
            vae,
            clip,
            image,
            supir_sampler,
            supir_model,
            fp8_unet,
            vae_tile_size,
            sampler_tile_size,
            positive_prompt,
            negative_prompt,
            steps,
            cfg_scale_start,
            cfg_scale_end,
            s_noise,
            control_scale_start,
            control_scale_end,
            seed,
            color_match,
        )

    def supir(
        self,
        model,
        vae,
        clip,
        image,
        supir_sampler,
        supir_model,
        fp8_unet,
        vae_tile_size,
        sampler_tile_size,
        positive_prompt,
        negative_prompt,
        steps,
        cfg_scale_start,
        cfg_scale_end,
        s_noise,
        control_scale_start,
        control_scale_end,
        seed,
        color_match,
    ):
        def round_to_even(n):
            return round(n / 2) * 2

        graph = Graph()

        use_tiled_vae = vae_tile_size != 0

        supir_model, supir_vae = graph.SUPIR_model_loader_v2(
            model,
            clip,
            vae,
            supir_model,
            fp8_unet=fp8_unet,
            diffusion_dtype="auto",
            high_vram=False,
        )

        denoised_image, denoised_latents = graph.SUPIR_first_stage(
            supir_vae,
            image,
            use_tiled_vae=use_tiled_vae,
            encoder_tile_size=vae_tile_size,
            decoder_tile_size=vae_tile_size,
            encoder_dtype="auto",
        )

        positive, negative = graph.SUPIR_conditioner(
            supir_model, denoised_latents, positive_prompt, negative_prompt
        )

        latent = graph.SUPIR_encode(
            supir_vae,
            denoised_image,
            use_tiled_vae,
            vae_tile_size,
            encoder_dtype="auto",
        )

        dpmpp_eta, edm_s_churn, restore_cfg, sampler = graph.SP_SupirSampler(supir_sampler)

        latent = graph.SUPIR_sample(
            supir_model,
            latent,
            positive,
            negative,
            seed,
            steps,
            cfg_scale_start,
            cfg_scale_end,
            edm_s_churn,
            s_noise,
            dpmpp_eta,
            control_scale_start,
            control_scale_end,
            restore_cfg,
            keep_model_loaded=False,
            sampler=sampler,
            sampler_tile_size=sampler_tile_size,
            sampler_tile_stride=round_to_even(sampler_tile_size / 2),
        )

        supir_image = graph.SUPIR_decode(
            supir_vae, latent, use_tiled_vae, vae_tile_size
        )

        if color_match:
            supir_image = graph.ImageColorMatchP(supir_image, image, color_space="LAB")

        return {
            "result": (supir_image, denoised_image),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "SP_SupirSampler": SP_SupirSampler,
    "SP_SupirSampler_DPMPP2M": SP_SupirSampler_DPMPP2M,
    "SP_SupirSampler_EDM": SP_SupirSampler_EDM,
    "SP_Supir": SP_Supir,
}
