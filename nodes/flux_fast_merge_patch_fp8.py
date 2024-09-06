from comfy.model_patcher import ModelPatcher, string_to_seed
import collections, logging, torch, comfy.utils
import inspect, binascii

CATEGORY = "SP-Nodes"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
    
class SP_FluxFastMergePatchFP8:
    CATEGORY = CATEGORY
    RETURN_TYPES = (AnyType("*"), )
    FUNCTION = "apply_patch"
    OUTPUT_NODE = True

    DESCRIPTION = """This node, after running, modifies the ComfyUI system function to speed up model merging/applying LoRA on the GPU. The speed can increase by up to two times.

To apply the patch, simply run this node.

Important:
- This patch is only useful for FP8 models.
- To UNDO the patch, you need to restart ComfyUI.
- This patch is useless if you are using GGUF-Unet models.
"""
    
    ORIGINAL_PWTD = None
    CRC_PWTD = 2604784964

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "info":("STRING", {"multiline": True,"default": SP_FluxFastMergePatchFP8.DESCRIPTION}),
            },
            "optional": {
                "optional":(AnyType("*"), ),
            }
        }

    def apply_patch(self, **kwargs):
        pwtd = ModelPatcher.patch_weight_to_device
        src = inspect.getsource(pwtd)
        crc = binascii.crc32(src.encode('utf-8'))

        is_original = SP_FluxFastMergePatchFP8.CRC_PWTD == crc 
        is_patched = pwtd.__name__ == 'patch_weight_to_device_v1'

        if not is_patched and not is_original:
            raise Exception(f'NAME: {pwtd.__name__} CRC: {crc} => Conflict with other nodes or outdated patch!')

        if is_patched:
            logging.info('[SP_FluxFastMergePatchFP8] already patched!')

        new_device = torch.device('cuda')
        store_device = torch.device('cpu')

        ### MonkeyPatch ###
        def patch_weight_to_device_v1(self, key, device_to=None, inplace_update=False):
            if key not in self.patches:
                    return

            weight = comfy.utils.get_attr(self.model, key)

            inplace_update = self.weight_inplace_update or inplace_update

            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

            temp_weight = weight.to(new_device).to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            out_weight = out_weight.to(weight.dtype).to(store_device)

            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

            del weight

        def patch_weight_to_device_wrap(s, key, device_to=None, inplace_update=False):
            if key not in s.patches:
                return

            w = comfy.utils.get_attr(s.model, key)
            w.to(new_device)
            dtype = w.dtype

            globals()['orig_mp'](s, key, device_to=None, inplace_update=inplace_update)

            if device_to is not None:
                comfy.utils.get_attr(s.model, key).to(dtype).to(store_device)


        if SP_FluxFastMergePatchFP8.ORIGINAL_PWTD == None and is_original:
            SP_FluxFastMergePatchFP8.ORIGINAL_PWTD = pwtd
            ModelPatcher.patch_weight_to_device = patch_weight_to_device_v1
            logging.info('[SP_FluxFastMergePatchFP8] patch succeeded!')


        return SP_FluxFastMergePatchFP8.ORIGINAL_PWTD != None, 

        

NODE_CLASS_MAPPINGS = {
    "SP_FluxFastMergePatchFP8 [Experimental]": SP_FluxFastMergePatchFP8,
}

