import sys
import nodes

NODE_CLASS_MAPPINGS = {}

if 'impact' in sys.modules:
    impact = sys.modules['impact']

    PixelKSampleHook=impact.hooks.PixelKSampleHook
    try_install_custom_node=impact.utils.try_install_custom_node

    class InjectNoiseHook(PixelKSampleHook):
        def __init__(self, source, seed, start_strength, end_strength):
            super().__init__()
            self.source = source
            self.seed = seed
            self.start_strength = start_strength
            self.end_strength = end_strength

        def post_encode(self, samples):
            cur_step = self.cur_step

            size = samples['samples'].shape
            seed = cur_step + self.seed + cur_step

            if "InjectLatentNoise+" in nodes.NODE_CLASS_MAPPINGS:
                InjectLatentNoise = nodes.NODE_CLASS_MAPPINGS["InjectLatentNoise+"]
            else:
                try_install_custom_node('https://github.com/cubiq/ComfyUI_essentials',
                                            "To use 'NoiseInjectionEssentialsHookProvider', 'ComfyUI_essentials' extension is required.")
                raise Exception("'InjectLatentNoise+' nodes are not installed.")

            # inj noise
            mask = None
            if 'noise_mask' in samples:
                mask = samples['noise_mask']

            strength = self.start_strength + (self.end_strength - self.start_strength) * cur_step / self.total_step
            samples = InjectLatentNoise().execute(samples, seed, strength, normalize="false", mask=mask)[0]
            print(f"[Impact Pack] InjectNoiseHook: strength = {strength}")

            if mask is not None:
                samples['noise_mask'] = mask

            return samples

    class NoiseInjectionEssentialsHookProvider:
        schedules = ["simple"]

        @classmethod
        def INPUT_TYPES(s):
            return {"required": {
                        "schedule_for_iteration": (s.schedules,),
                        "source": (["CPU", "GPU"],),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 200.0, "step": 0.01}),
                        "end_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 200.0, "step": 0.01}),
                        },
                    }

        RETURN_TYPES = ("PK_HOOK",)
        FUNCTION = "doit"

        CATEGORY = "ImpactPack/Upscale"

        def doit(self, schedule_for_iteration, source, seed, start_strength, end_strength):
            try:
                hook = None
                if schedule_for_iteration == "simple":
                    hook = InjectNoiseHook(source, seed, start_strength, end_strength)

                return (hook, )
            except Exception as e:
                print("[ERROR] NoiseInjectionHookProvider: 'ComfyUI Noise' custom node isn't installed. You must install 'BlenderNeko/ComfyUI Noise' extension to use this node.")
                print(f"\t{e}")
                pass

    NODE_CLASS_MAPPINGS = {
        "NoiseInjectionEssentialsHookProvider": NoiseInjectionEssentialsHookProvider,
    }
