import folder_paths, nodes, logging, comfy
from comfy_execution.graph import DynamicPrompt

from .Graph import Graph


class SP_FluxUnsampler_InverseSampler:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                "steps": (
                    "INT",
                    {
                        "default": 60,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "use_noise_mixer": ("BOOLEAN", {"default": False}),
                "mix_percent": (
                    "FLOAT",
                    {"default": 0.98, "min": 0, "max": 1.0, "step": 0.01},
                ),
                "random_noise": (
                    "FLOAT",
                    {"default": 0.0, "min": 0, "max": 100.0, "step": 0.01},
                ),
                "mix_type": (["mix", "add"], {'default': 'mix'}),
                "random_mix_type": (["mix", "add"], {'default': 'add'}),
                "take_diff": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = ("flux_unsampler_sampler",)
    FUNCTION = "fn"

    def fn(self, **kwargs):
        return None,


class SP_FluxUnsampler_ForwardODESampler:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "gamma": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "the paper leaves this at 0.5",
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "the strength of the guidance. The paper does not decrease this below 0.7",
                    },
                ),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": 'the step that the sampler starts guiding the sampling towards the image in "latent_image"',
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 9,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "the last step for guiding the sampling (not inclusive)",
                    },
                ),
                "eta_trend": (
                    ["constant", "linear_increase", "linear_decrease"],
                    {
                        "tooltip": "how the eta should increase/decrease/stay constant between start_step and end_step",
                    },
                ),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("flux_unsampler_sampler",)
    FUNCTION = "fn"

    def fn(self, **kwargs):
        return None,

class SP_FluxUnsampler:
    CATEGORY = "SP-Nodes/Group Nodes"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": ("MODEL", {"rawLink": True}),
                "clip": ("CLIP", {"rawLink": True}),
                "vae": ("VAE", {"rawLink": True}),
                "image": ("IMAGE", {"rawLink": True}),
                "flux_unsampler_sampler": ("flux_unsampler_sampler", {"rawLink": True}),
                "positive_unsampler": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "placeholder": "positive_unsampler",
                    },
                ),
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "placeholder": "positive",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 28,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "guidance": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "max_shift": (
                    "FLOAT",
                    {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "base_shift": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image.",
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
            },
            "hidden": {
				"prompt": "PROMPT",
				"id": "UNIQUE_ID",
				"workflow": "EXTRA_PNGINFO",
				"dynprompt": "DYNPROMPT",
			}
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "fn"

    def fn(
        self,
        model,
        clip,
        vae,
        image,
        flux_unsampler_sampler,
        positive_unsampler,
        positive,
        steps,
        guidance,
        max_shift,
        base_shift,
        scheduler,
        tile_size,
        vae_decode,
        preview,
        prompt, id, workflow, dynprompt: DynamicPrompt
    ):
        graph = Graph()

        # get image size
        width, height, img_count = graph.GetImageSize(image)

        # parse samplers inputs
        flux_unsampler_sampler = dynprompt.get_node(node_id=flux_unsampler_sampler[0])
        inputs = flux_unsampler_sampler['inputs']
        class_type = flux_unsampler_sampler['class_type']
        is_ode_sampler = class_type == 'SP_FluxUnsampler_ForwardODESampler'

        print(f'is_ode_sampler: {is_ode_sampler}, class_type: {class_type}, inputs: {inputs}')

        reverse_ode = is_ode_sampler
        sampler = graph.FluxForwardODESampler(gamma=inputs['gamma'], seed=inputs['seed']) if is_ode_sampler else graph.FluxInverseSampler()

        latent = graph.VAEEncode(image, vae, tile_size=tile_size)

        def unsampling(model):
            cond = graph.CLIPTextEncode(clip, positive_unsampler)
            cond = graph.FluxDeGuidance(cond, guidance=0)

            sched = graph.BasicScheduler(
                model, scheduler=scheduler, steps=steps if is_ode_sampler else inputs['steps'], denoise=1.0
            )
            sched = graph.FlipSigmas(sched)

            model2 = graph.InFluxModelSamplingPred(
                model, width, height, max_shift=max_shift, base_shift=base_shift
            )
            guider = graph.BasicGuider(model2, cond)

            output, denoised_output = graph.SamplerCustomAdvanced(
                graph.DisableNoise(), guider, sampler, sched, latent
            )
            return output

        def sampling(model, unsampling_latent):
            cond = graph.CLIPTextEncode(clip, positive)
            cond = graph.FluxDeGuidance(cond, guidance=guidance)
            model = graph.OutFluxModelSamplingPred(
                model,
                width,
                height,
                max_shift=max_shift,
                base_shift=base_shift,
                reverse_ode=reverse_ode,
            )
            guider = graph.BasicGuider(model, cond)

            sched = graph.BasicScheduler(
                model, scheduler=scheduler, steps=steps, denoise=1.0
            )

            # get valid sampler
            sampler = None
            if is_ode_sampler:
                sampler = graph.FluxReverseODESampler(
                    model,
                    latent,
                    eta=inputs['eta'],
                    start_step=inputs['start_step'],
                    end_step=inputs['end_step'],
                    eta_trend=inputs['eta_trend'],
                )
            else:
                sampler = graph.KSamplerSelect(sampler_name=inputs['sampler_name'])

            # apply noise mixer if needed
            if not is_ode_sampler and inputs['use_noise_mixer']:
                unsampling_latent = graph.FluxNoiseMixer(latent, unsampling_latent, inputs['mix_percent'], inputs['random_noise'], inputs['mix_type'], inputs['random_mix_type'], inputs['take_diff'])

            output, denoised_output = graph.SamplerCustomAdvanced(
                graph.DisableNoise(), guider, sampler, sched, unsampling_latent
            )
            return output

        output = unsampling(model)
        output = sampling(model, output)

        image = None
        if vae_decode or preview:
            image = graph.VAEDecode(output, vae, tile_size)

            if preview:
                graph.PreviewImage(image)

        return {
            "result": (output, image),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "SP_FluxUnsampler_InverseSampler": SP_FluxUnsampler_InverseSampler,
    "SP_FluxUnsampler_ForwardODESampler": SP_FluxUnsampler_ForwardODESampler,
    "SP_FluxUnsampler": SP_FluxUnsampler,
}
