import nodes
from functools import wraps
from comfy_execution.graph_utils import GraphBuilder

RGTHREE = ('rgthree-comfy', 'https://github.com/rgthree/rgthree-comfy')
IMPACT_PACK = ('ComfyUI-Impact-Pack', 'https://github.com/ltdrdata/ComfyUI-Impact-Pack')
NF4_LORA_LOADER = ('ComfyUI_bitsandbytes_NF4-Lora', 'https://github.com/bananasss00/ComfyUI_bitsandbytes_NF4-Lora')
GGUF = ('ComfyUI-GGUF', 'https://github.com/city96/ComfyUI-GGUF')
SUPIR = ('ComfyUI-SUPIR', 'https://github.com/kijai/ComfyUI-SUPIR')
ESSENTIALS = ('ComfyUI_essentials', 'https://github.com/cubiq/ComfyUI_essentials')
FLORENCE2 = ('ComfyUI-Florence2', 'https://github.com/kijai/ComfyUI-Florence2')
FLUXTAPOZ = ('ComfyUI-Fluxtapoz', 'https://github.com/logtd/ComfyUI-Fluxtapoz')

FOR_CHECK_NODES = []

def get_requirements():
    return {
        'SP_Supir': ["ComfyUI-SUPIR", "ComfyUI_essentials"],
        'SP_SDKSampler': ["rgthree-comfy", "ComfyUI-Impact-Pack"],
        'SP_FluxKSampler': ["rgthree-comfy", "ComfyUI-Impact-Pack"],
        'SP_FluxLoader': ["ComfyUI_bitsandbytes_NF4-Lora", "ComfyUI-GGUF"],
        'SP_FlorenceCaption': ["ComfyUI-Florence2"],
        'SP_FluxUnsampler': ["ComfyUI-Fluxtapoz", "ComfyUI_essentials"],
    }

def get_missing_nodes():
    missing_nodes = {}
    for node_name, extension_name, install_url in FOR_CHECK_NODES:
        if node_name not in nodes.NODE_CLASS_MAPPINGS:
            if extension_name not in missing_nodes:
                missing_nodes[extension_name] = {'install_url': install_url, 'nodes': []}

            missing_nodes[extension_name]['nodes'].append(node_name)
            # print(f'missing node: {node_name}')
    return missing_nodes


def requires_extension(node_name, extension_name, install_url):
    
    def decorator(func):
        FOR_CHECK_NODES.append((node_name, extension_name, install_url))
        return func
    
    return decorator

class Graph:
    def __init__(self):
        self.graph = GraphBuilder()

    def finalize(self):
        return self.graph.finalize()

    def lookup_node(self, id):
        return self.graph.lookup_node(id)

    def ConditioningZeroOut(self, conditioning):
        '''
        return conditioning
        '''
        node = self.graph.node('ConditioningZeroOut', conditioning=conditioning)
        return node.out(0)

    def ConditioningSetTimestepRange(self, conditioning, start, end):
        '''
        return conditioning
        '''
        node = self.graph.node('ConditioningSetTimestepRange', conditioning=conditioning, start=start, end=end)
        return node.out(0)

    def ConditioningCombine(self, conditioning_1, conditioning_2):
        '''
        return conditioning
        '''
        node = self.graph.node('ConditioningCombine', conditioning_1=conditioning_1, conditioning_2=conditioning_2)
        return node.out(0)


    @requires_extension('Any Switch (rgthree)', *RGTHREE)
    def AnySwitch(self, any_01=None, any_02=None, any_03=None, any_04=None, any_05=None):
        node = self.graph.node('Any Switch (rgthree)', any_01=any_01, any_02=any_02, any_03=any_03, any_04=any_04, any_05=any_05)
        return node.out(0)

    @requires_extension('ImpactIfNone', *IMPACT_PACK)
    def ImpactIfNone(self, signal=None, any_input=None):
        '''
        return signal, bool (if any_input not None=True)
        '''
        node = self.graph.node('ImpactIfNone', signal=signal, any_input=any_input)
        return node.out(0), node.out(1)

    @requires_extension('ImpactConditionalBranch', *IMPACT_PACK)
    def ImpactConditionalBranch(self, tt_value, ff_value, cond):
        '''
        return selected_value
        '''
        node = self.graph.node('ImpactConditionalBranch', tt_value=tt_value, ff_value=ff_value, cond=cond)
        return node.out(0)
    
    @requires_extension('ImpactCompare', *IMPACT_PACK)
    def ImpactCompare(self, a, b, cmp=r"a = b"):
        '''
        return boolean
        '''
        node = self.graph.node('ImpactCompare', a=a, b=b, cmp=cmp)
        return node.out(0)

    @requires_extension('ImpactBoolean', *IMPACT_PACK)
    def ImpactBoolean(self, value=True):
        '''
        return boolean
        '''
        node = self.graph.node('ImpactBoolean', value=value)
        return node.out(0)

    @requires_extension('ImpactLogicalOperators', *IMPACT_PACK)
    def ImpactLogicalOperators(self, bool_a, bool_b, operator=r"and"):
        '''
        return boolean
        '''
        node = self.graph.node('ImpactLogicalOperators', bool_a=bool_a, bool_b=bool_b, operator=operator)
        return node.out(0)


    def CheckpointLoaderSimple(self, ckpt_name):
        load_checkpoint = self.graph.node("CheckpointLoaderSimple", ckpt_name=ckpt_name)
        return load_checkpoint.out(0), load_checkpoint.out(1), load_checkpoint.out(2)

    def VAELoader(self, vae_name):
        load_vae = self.graph.node("VAELoader", vae_name=vae_name)
        return load_vae.out(0)

    def LoraLoader(self, model, clip, lora_name, lora_strength):
        node = self.graph.node(
            "LoraLoader",
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        return node.out(0), node.out(1)

    def CLIPSetLastLayer(self, clip, stop_at_clip_layer):
        clip_set_last_layer = self.graph.node(
            "CLIPSetLastLayer", clip=clip, stop_at_clip_layer=stop_at_clip_layer
        )
        return clip_set_last_layer.out(0)

    def CLIPTextEncode(self, clip, text):
        return self.graph.node("CLIPTextEncode", clip=clip, text=text).out(0)

    def EmptyLatentImage(self, width, height, batch_size):
        return self.graph.node(
            "EmptyLatentImage",
            width=width,
            height=height,
            batch_size=batch_size,
        ).out(0)

    def EmptySD3LatentImage(self, width, height, batch_size):
        return self.graph.node(
            "EmptySD3LatentImage",
            width=width,
            height=height,
            batch_size=batch_size,
        ).out(0)

    def SP_Pipe(
        self,
        sp_pipe,
        model=None,
        clip=None,
        vae=None,
        positive=None,
        negative=None,
        latent=None,
        image=None,
    ):
        '''
        return sp_pipe, model, clip, vae, positive, negative, latent, image
        '''
        node = self.graph.node(
            "SP_Pipe",
            sp_pipe=sp_pipe,
            model=model,
            clip=clip,
            vae=vae,
            positive=positive,
            negative=negative,
            latent=latent,
            image=image,
        )
        return tuple(node.out(i) for i in range(8))

    def VAEEncode(self, image, vae, tile_size=0):
        node = self.graph.node(
            "VAEEncodeTiled" if tile_size else "VAEEncode",
            pixels=image,
            vae=vae,
            tile_size=tile_size,
        )
        return node.out(0)

    def VAEDecode(self, latent, vae, tile_size=0):
        node = self.graph.node(
            "VAEDecodeTiled" if tile_size else "VAEDecode",
            samples=latent,
            vae=vae,
            tile_size=tile_size,
        )
        return node.out(0)

    def KSampler(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
    ):
        node = self.graph.node(
            "KSampler",
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )
        return node.out(0)

    def PreviewImage(self, image):
        self.graph.node("PreviewImage", images=image)

    def UNETLoader(self, unet_name, weight_dtype):
        return self.graph.node(
            "UNETLoader", unet_name=unet_name, weight_dtype=weight_dtype
        ).out(0)

    @requires_extension("SP_UnetLoaderBNB", *NF4_LORA_LOADER)
    def SP_UnetLoaderBNB(self, unet_name, load_dtype):
        return self.graph.node(
            "SP_UnetLoaderBNB",
            unet_name=unet_name,
            load_dtype=load_dtype,
        ).out(0)

    @requires_extension("UnetLoaderGGUF", *GGUF)
    def UnetLoaderGGUF(self, unet_name):
        return self.graph.node("UnetLoaderGGUF", unet_name=unet_name).out(0)

    @requires_extension("DualCLIPLoaderGGUF", *GGUF)
    def CLIPLoaderGGUF(self, clip_name, type=r"sd3"):
        '''
        return clip
        '''
        node = self.graph.node('CLIPLoaderGGUF', clip_name=clip_name, type=type)
        return node.out(0)

    @requires_extension("DualCLIPLoaderGGUF", *GGUF)
    def DualCLIPLoaderGGUF(self, clip_name1, clip_name2, type):
        return self.graph.node(
            "DualCLIPLoaderGGUF",
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            type=type,
        ).out(0)

    @requires_extension("TripleCLIPLoaderGGUF", *GGUF)
    def TripleCLIPLoaderGGUF(self, clip_name1, clip_name2, clip_name3):
        '''
        return clip
        '''
        node = self.graph.node('TripleCLIPLoaderGGUF', clip_name1=clip_name1, clip_name2=clip_name2, clip_name3=clip_name3)
        return node.out(0)


    def FluxGuidance(self, conditioning, guidance):
        return self.graph.node(
            "FluxGuidance", conditioning=conditioning, guidance=guidance
        ).out(0)

    def ModelSamplingFlux(self, model, max_shift, base_shift, width, height):
        return self.graph.node(
            "ModelSamplingFlux",
            model=model,
            max_shift=max_shift,
            base_shift=base_shift,
            width=width,
            height=height,
        ).out(0)
    
    @requires_extension('SP_SupirSampler', *SUPIR)
    def SP_SupirSampler(self, supir_sampler):
        '''
        return dpmpp_eta, edm_s_churn, restore_cfg, sampler
        '''
        node = self.graph.node('SP_SupirSampler', supir_sampler=supir_sampler)
        return node.out(0), node.out(1), node.out(2), node.out(3)
    
    @requires_extension('SUPIR_model_loader_v2', *SUPIR)
    def SUPIR_model_loader_v2(self, model, clip, vae, supir_model, fp8_unet, diffusion_dtype, high_vram):
        '''
        return supir_model, supir_vae
        '''
        node = self.graph.node('SUPIR_model_loader_v2', model=model, clip=clip, vae=vae, supir_model=supir_model, fp8_unet=fp8_unet, diffusion_dtype=diffusion_dtype, high_vram=high_vram)
        return node.out(0), node.out(1)
    
    @requires_extension('SUPIR_first_stage', *SUPIR)
    def SUPIR_first_stage(self, SUPIR_VAE, image, use_tiled_vae, encoder_tile_size, decoder_tile_size, encoder_dtype):
        '''
        return denoised_image, denoised_latents
        '''
        node = self.graph.node('SUPIR_first_stage', SUPIR_VAE=SUPIR_VAE, image=image, use_tiled_vae=use_tiled_vae, encoder_tile_size=encoder_tile_size, decoder_tile_size=decoder_tile_size, encoder_dtype=encoder_dtype)
        return node.out(1), node.out(2)
    
    @requires_extension('SUPIR_conditioner', *SUPIR)
    def SUPIR_conditioner(self, SUPIR_model, latents, positive_prompt, negative_prompt):
        '''
        return positive, negative
        '''
        node = self.graph.node('SUPIR_conditioner', SUPIR_model=SUPIR_model, latents=latents, captions='', positive_prompt=positive_prompt, negative_prompt=negative_prompt)
        return node.out(0), node.out(1)
    
    @requires_extension('SUPIR_encode', *SUPIR)
    def SUPIR_encode(self, SUPIR_VAE, image, use_tiled_vae, encoder_tile_size, encoder_dtype):
        '''
        return latent
        '''
        node = self.graph.node('SUPIR_encode', SUPIR_VAE=SUPIR_VAE, image=image, use_tiled_vae=use_tiled_vae, encoder_tile_size=encoder_tile_size, encoder_dtype=encoder_dtype)
        return node.out(0)
    
    @requires_extension('SUPIR_sample', *SUPIR)
    def SUPIR_sample(self, SUPIR_model, latents, positive, negative, seed, steps, cfg_scale_start, cfg_scale_end, EDM_s_churn, s_noise, DPMPP_eta, control_scale_start, control_scale_end, restore_cfg, keep_model_loaded, sampler, sampler_tile_size, sampler_tile_stride):
        '''
        return latent
        '''
        node = self.graph.node('SUPIR_sample', SUPIR_model=SUPIR_model, latents=latents, positive=positive, negative=negative, seed=seed, steps=steps, cfg_scale_start=cfg_scale_start, cfg_scale_end=cfg_scale_end, EDM_s_churn=EDM_s_churn, s_noise=s_noise, DPMPP_eta=DPMPP_eta, control_scale_start=control_scale_start, control_scale_end=control_scale_end, restore_cfg=restore_cfg, keep_model_loaded=keep_model_loaded, sampler=sampler, sampler_tile_size=sampler_tile_size, sampler_tile_stride=sampler_tile_stride)
        return node.out(0)
    
    @requires_extension('SUPIR_decode', *SUPIR)
    def SUPIR_decode(self, SUPIR_VAE, latents, use_tiled_vae, decoder_tile_size):
        '''
        return image
        '''
        node = self.graph.node('SUPIR_decode', SUPIR_VAE=SUPIR_VAE, latents=latents, use_tiled_vae=use_tiled_vae, decoder_tile_size=decoder_tile_size)
        return node.out(0)
    
    @requires_extension('ImageColorMatch+', *ESSENTIALS)
    def ImageColorMatchP(self, image, reference, reference_mask=None, color_space="LAB", factor=1, device="auto", batch_size=0):
        '''
        return image
        '''
        node = self.graph.node('ImageColorMatch+', image=image, reference=reference, reference_mask=reference_mask, color_space=color_space, factor=factor, device=device, batch_size=batch_size)
        return node.out(0)
    
    @requires_extension('GetImageSize+', *ESSENTIALS)
    def GetImageSize(self, image):
        '''
        return width, height, count
        '''
        node = self.graph.node('GetImageSize+', image=image)
        return node.out(0), node.out(1), node.out(2)

    @requires_extension('InjectLatentNoise+', *ESSENTIALS)
    def InjectLatentNoise(self, latent, mask=None, noise_seed=0, noise_strength=1, normalize=r"false"):
        '''
        return latent
        '''
        node = self.graph.node('InjectLatentNoise+', latent=latent, mask=mask, noise_seed=noise_seed, noise_strength=noise_strength, normalize=normalize)
        return node.out(0)

    
    @requires_extension('Florence2Run', *FLORENCE2)
    def Florence2Run(self, image, florence2_model, text_input=r"", task=r"more_detailed_caption", fill_mask=True, keep_model_loaded=False, max_new_tokens=1024, num_beams=3, do_sample=True, output_mask_select=r"", seed=1):
        '''
        return image, mask, caption, data
        '''
        node = self.graph.node('Florence2Run', image=image, florence2_model=florence2_model, text_input=text_input, task=task, fill_mask=fill_mask, keep_model_loaded=keep_model_loaded, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample, output_mask_select=output_mask_select, seed=seed)
        return node.out(0), node.out(1), node.out(2), node.out(3)

    @requires_extension('DownloadAndLoadFlorence2Model', *FLORENCE2)
    def DownloadAndLoadFlorence2Model(self, lora, model=r"microsoft/Florence-2-base", precision=r"fp16", attention=r"sdpa"):
        '''
        return florence2_model
        '''
        node = self.graph.node('DownloadAndLoadFlorence2Model', lora=lora, model=model, precision=precision, attention=attention)
        return node.out(0)

    def SamplerCustomAdvanced(self, noise, guider, sampler, sigmas, latent_image):
        '''
        return output, denoised_output
        '''
        node = self.graph.node('SamplerCustomAdvanced', noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent_image)
        return node.out(0), node.out(1)

    def BasicGuider(self, model, conditioning):
        '''
        return guider
        '''
        node = self.graph.node('BasicGuider', model=model, conditioning=conditioning)
        return node.out(0)

    def BasicScheduler(self, model, scheduler=r"simple", steps=28, denoise=1):
        '''
        return sigmas
        '''
        node = self.graph.node('BasicScheduler', model=model, scheduler=scheduler, steps=steps, denoise=denoise)
        return node.out(0)

    def DisableNoise(self):
        '''
        return noise
        '''
        node = self.graph.node('DisableNoise')
        return node.out(0)

    def FlipSigmas(self, sigmas):
        '''
        return sigmas
        '''
        node = self.graph.node('FlipSigmas', sigmas=sigmas)
        return node.out(0)

    @requires_extension('InFluxModelSamplingPred', *FLUXTAPOZ)
    def InFluxModelSamplingPred(self, model, width, height, max_shift=1.15, base_shift=0.5):
        '''
        return model
        '''
        node = self.graph.node('InFluxModelSamplingPred', model=model, width=width, height=height, max_shift=max_shift, base_shift=base_shift)
        return node.out(0)

    @requires_extension('FluxDeGuidance', *FLUXTAPOZ)
    def FluxDeGuidance(self, conditioning, guidance:float=0):
        '''
        return conditioning
        '''
        node = self.graph.node('FluxDeGuidance', conditioning=conditioning, guidance=guidance)
        return node.out(0)

    @requires_extension('FluxForwardODESampler', *FLUXTAPOZ)
    def FluxForwardODESampler(self, gamma=0.5, seed=0):
        '''
        return sampler
        '''
        node = self.graph.node('FluxForwardODESampler', gamma=gamma, seed=seed)
        return node.out(0)

    @requires_extension('OutFluxModelSamplingPred', *FLUXTAPOZ)
    def OutFluxModelSamplingPred(self, model, width, height, max_shift=1.15, base_shift=0.5, reverse_ode=True):
        '''
        return model
        '''
        node = self.graph.node('OutFluxModelSamplingPred', model=model, width=width, height=height, max_shift=max_shift, base_shift=base_shift, reverse_ode=reverse_ode)
        return node.out(0)

    @requires_extension('FluxReverseODESampler', *FLUXTAPOZ)
    def FluxReverseODESampler(self, model, latent_image, eta=0.9, start_step=0, end_step=9, eta_trend=r"constant"):
        '''
        return sampler
        '''
        node = self.graph.node('FluxReverseODESampler', model=model, latent_image=latent_image, eta=eta, start_step=start_step, end_step=end_step, eta_trend=eta_trend)
        return node.out(0)

    @requires_extension('FluxNoiseMixer', *FLUXTAPOZ)
    def FluxNoiseMixer(self, latent, noise, mix_percent=0.98, random_noise=0, mix_type=r"mix", random_mix_type=r"add", take_diff=False):
        '''
        return latent
        '''
        node = self.graph.node('FluxNoiseMixer', latent=latent, noise=noise, mix_percent=mix_percent, random_noise=random_noise, mix_type=mix_type, random_mix_type=random_mix_type, take_diff=take_diff)
        return node.out(0)

    @requires_extension('FluxInverseSampler', *FLUXTAPOZ)
    def FluxInverseSampler(self):
        '''
        return sampler
        '''
        node = self.graph.node('FluxInverseSampler')
        return node.out(0)

    def KSamplerSelect(self, sampler_name=r"dpmpp_2m"):
        '''
        return sampler
        '''
        node = self.graph.node('KSamplerSelect', sampler_name=sampler_name)
        return node.out(0)
    
    @requires_extension('ConsoleDebug+', *ESSENTIALS)
    def ConsoleDebug(self, value, prefix=r"Value:"):
        node = self.graph.node('ConsoleDebug+', value=value, prefix=prefix)
