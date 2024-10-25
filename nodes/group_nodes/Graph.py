from comfy_execution.graph_utils import GraphBuilder


class Graph:
    def __init__(self):
        self.graph = GraphBuilder()

    def finalize(self):
        return self.graph.finalize()

    def lookup_node(self, id):
        return self.graph.lookup_node(id)

    def AnySwitch(self, any_01=None, any_02=None, any_03=None, any_04=None, any_05=None):
        # dep: rgthree
        node = self.graph.node('Any Switch (rgthree)', any_01=any_01, any_02=any_02, any_03=any_03, any_04=any_04, any_05=any_05)
        return node.out(0)

    def ImpactIfNone(self, signal=None, any_input=None):
        # dep: impactpack
        '''
        return signal, bool (if any_input not None=True)
        '''
        node = self.graph.node('ImpactIfNone', signal=signal, any_input=any_input)
        return node.out(0), node.out(1)

    def ImpactConditionalBranch(self, tt_value, ff_value, cond):
        # dep: impactpack
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

    def SP_UnetLoaderBNB(self, unet_name, load_dtype):
        # dep: nf4 loader lora
        return self.graph.node(
            "SP_UnetLoaderBNB",
            unet_name=unet_name,
            load_dtype=load_dtype,
        ).out(0)

    def UnetLoaderGGUF(self, unet_name):
        # dep: gguf
        return self.graph.node("UnetLoaderGGUF", unet_name=unet_name).out(0)

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