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

FOR_CHECK_NODES = []

def get_requirements():
    return {
        'SP_Supir': ["ComfyUI-SUPIR", "ComfyUI_essentials"],
        'SP_SDKSampler': ["rgthree-comfy", "ComfyUI-Impact-Pack"],
        'SP_FluxKSampler': ["rgthree-comfy", "ComfyUI-Impact-Pack"],
        'SP_FluxLoader': ["ComfyUI_bitsandbytes_NF4-Lora", "ComfyUI-GGUF"],
        'SP_FlorenceCaption': ["ComfyUI-Florence2"],
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
    def DualCLIPLoaderGGUF(self, clip_name1, clip_name2, type):
        return self.graph.node(
            "DualCLIPLoaderGGUF",
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            type=type,
        ).out(0)

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
