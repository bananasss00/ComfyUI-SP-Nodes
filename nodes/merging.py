import math
import random
import hashlib
import numpy as np
import torch
import time
import io

import comfy.utils
import comfy.model_management

from comfy_extras.nodes_model_merging import ModelMergeBlocks
from comfy.model_detection import count_blocks

def random_scale_blocked(seed=None):
    if seed:
        random.seed(seed)
    
    center_values = [ random.random(), random.random(), random.random() ]
    presets = []
    input_blocks = 12
    middle_blocks = 3
    out_blocks = 12
    
    for i in range(25):
        if i < input_blocks:
            if i == 5:
                center_value = center_values[0]
                t = 0.5
            else:
                center_value = center_values[0]
                t = i / 5
        elif i < input_blocks + middle_blocks:
            if i == 12:
                center_value = center_values[1]
                t = 0.5
            else:
                center_value = center_values[1]
                t = (i - 6) / 5
        else:
            if i == 21:
                center_value = center_values[2]
                t = 0.5
            else:
                center_value = center_values[2]
                t = (i - 15) / 5
        
        value = center_value + (random.random() - 0.5) * (1 - abs(2 * t - 1))
        presets.append(value)
    
    return presets

class GodnessMerger_Layer:
    FUNCTION = "merge"
    CATEGORY = "SP-Nodes/model_merging"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  
                "min": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "type": (["custom", "random"], ),
                "custom_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def merge(self, seed, min, max, type, custom_value, **kwargs):
        raise NotImplementedError()
    
    def update_custom_value(self, value_type, value_custom, min, max):
        if value_type == 'random':
            return random.uniform(min, max)
        
        return value_custom

class GodnessMerger_LayerExperimental:
    FUNCTION = "merge"
    CATEGORY = "SP-Nodes/model_merging"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  
                "min": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "type": (["custom", "random"], ),
                "custom_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def merge(self, model, seed, min, max, type, custom_value, **kwargs):
        raise NotImplementedError()
    
    def update_custom_value(self, value_type, value_custom, min, max):
        if value_type == 'random':
            return random.uniform(min, max)
        
        return value_custom
    
class GodnessMerger_TimeEmbed(GodnessMerger_Layer):
    RETURN_TYPES = ("GM_TIME_EMBED",)
   
    def merge(self, seed, min, max, type, custom_value, **kwargs):
        random.seed(1000 + seed)
        
        kwargs['time_embed.'] = self.update_custom_value(type, custom_value, min, max)
        
        return (kwargs, )
    
class GodnessMerger_LabelEmb(GodnessMerger_Layer):
    RETURN_TYPES = ("GM_LABEL_EMB",)
   
    def merge(self, seed, min, max, type, custom_value, **kwargs):
        random.seed(2000 + seed)
        
        kwargs['label_emb.'] = self.update_custom_value(type, custom_value, min, max)
        
        return (kwargs, )
    
class GodnessMerger_InputBlocks(GodnessMerger_Layer):
    RETURN_TYPES = ("GM_INPUT_BLOCKS",)
   
    def merge(self, seed, min, max, type, custom_value, **kwargs):
        random.seed(3000 + seed)
        
        for i in range(12):
            kwargs[f"input_blocks.{i}."] = self.update_custom_value(type, custom_value, min, max)
        
        return (kwargs, )
    
class GodnessMerger_InputBlocksExperimental(GodnessMerger_LayerExperimental):
    RETURN_TYPES = ("GM_INPUT_BLOCKS",)

    def merge(self, model, seed, min, max, type, custom_value, **kwargs):
        random.seed(3000 + seed)
        
        keys = model.model_state_dict().keys()
        
        def organize_keys(keys):
            blocks = {i: {'root': [], 'transformer_blocks': {j: [] for j in range(10)}} for i in range(12)}

            for key in keys:
                for i in range(12):
                    if key.startswith(f'diffusion_model.input_blocks.{i}'):
                        if 'transformer_blocks' in key:
                            for j in range(10):
                                if f'transformer_blocks.{j}.' in key:
                                    blocks[i]['transformer_blocks'][j].append(key[len("diffusion_model."):])
                        else:
                            blocks[i]['root'].append(key[len("diffusion_model."):])

            return blocks

        keys = organize_keys(keys)
        
        for layer in keys:
            root_ratio = self.update_custom_value(type, custom_value, min, max)
            for root in keys[layer]['root']:
                kwargs[root] = root_ratio
            
            for transformer_block in keys[layer]['transformer_blocks']:
                transformer_ratio = self.update_custom_value(type, custom_value, min, max)
                blocks = keys[layer]['transformer_blocks'][transformer_block]
                for block in blocks:
                    kwargs[block] = transformer_ratio

        return (kwargs, )
    
class GodnessMerger_MiddleBlock(GodnessMerger_Layer):
    RETURN_TYPES = ("GM_MIDDLE_BLOCK",)
   
    def merge(self, seed, min, max, type, custom_value, **kwargs):
        random.seed(4000 + seed)
        
        kwargs['middle_block.'] = self.update_custom_value(type, custom_value, min, max)
        
        return (kwargs, )
    
class GodnessMerger_MiddleBlockExperimental(GodnessMerger_LayerExperimental):
    RETURN_TYPES = ("GM_MIDDLE_BLOCK",)

    def merge(self, model, seed, min, max, type, custom_value, **kwargs):
        random.seed(4000 + seed)
        
        keys = model.model_state_dict().keys()
        
        def organize_keys(keys):
            blocks = {i: {'root': [], 'transformer_blocks': {j: [] for j in range(10)}} for i in range(12)}

            for key in keys:
                for i in range(12):
                    if key.startswith(f'diffusion_model.middle_block.{i}'):
                        if 'transformer_blocks' in key:
                            for j in range(10):
                                if f'transformer_blocks.{j}.' in key:
                                    blocks[i]['transformer_blocks'][j].append(key[len("diffusion_model."):])
                        else:
                            blocks[i]['root'].append(key[len("diffusion_model."):])

            return blocks

        keys = organize_keys(keys)
        
        for layer in keys:
            root_ratio = self.update_custom_value(type, custom_value, min, max)
            for root in keys[layer]['root']:
                kwargs[root] = root_ratio
            
            for transformer_block in keys[layer]['transformer_blocks']:
                transformer_ratio = self.update_custom_value(type, custom_value, min, max)
                blocks = keys[layer]['transformer_blocks'][transformer_block]
                for block in blocks:
                    kwargs[block] = transformer_ratio

        return (kwargs, )

class GodnessMerger_OutputBlocks(GodnessMerger_Layer):
    RETURN_TYPES = ("GM_OUTPUT_BLOCKS",)
   
    def merge(self, seed, min, max, type, custom_value, **kwargs):
        random.seed(5000 + seed)
        
        for i in range(12):
            kwargs[f"output_blocks.{i}."] = self.update_custom_value(type, custom_value, min, max)
        
        return (kwargs, )

class GodnessMerger_OutputBlocksExperimental(GodnessMerger_LayerExperimental):
    RETURN_TYPES = ("GM_OUTPUT_BLOCKS",)
    
    def merge(self, model, seed, min, max, type, custom_value, **kwargs):
        random.seed(5000 + seed)
        
        keys = model.model_state_dict().keys()
        
        def organize_keys(keys):
            blocks = {i: {'root': [], 'transformer_blocks': {j: [] for j in range(10)}} for i in range(12)}

            for key in keys:
                for i in range(12):
                    if key.startswith(f'diffusion_model.output_blocks.{i}'):
                        if 'transformer_blocks' in key:
                            for j in range(10):
                                if f'transformer_blocks.{j}.' in key:
                                    blocks[i]['transformer_blocks'][j].append(key[len("diffusion_model."):])
                        else:
                            blocks[i]['root'].append(key[len("diffusion_model."):])

            return blocks

        keys = organize_keys(keys)
        
        for layer in keys:
            root_ratio = self.update_custom_value(type, custom_value, min, max)
            for root in keys[layer]['root']:
                kwargs[root] = root_ratio
            
            for transformer_block in keys[layer]['transformer_blocks']:
                transformer_ratio = self.update_custom_value(type, custom_value, min, max)
                blocks = keys[layer]['transformer_blocks'][transformer_block]
                for block in blocks:
                    kwargs[block] = transformer_ratio

        return (kwargs, )
    
class GodnessMerger_Out(GodnessMerger_Layer):
    RETURN_TYPES = ("GM_OUT",)
   
    def merge(self, seed, min, max, type, custom_value, **kwargs):
        random.seed(6000 + seed)
        
        kwargs['out.'] = self.update_custom_value(type, custom_value, min, max)
        
        return (kwargs, )
    
class GodnessMerger_Apply:
    RETURN_TYPES = ("MODEL","STRING")
    RETURN_NAMES = ("MODEL", "RATIO_VALUES")
    FUNCTION = "merge"
    CATEGORY = "SP-Nodes/model_merging"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "time_embed": ("GM_TIME_EMBED",),
                "label_emb": ("GM_LABEL_EMB",),
                "input_blocks": ("GM_INPUT_BLOCKS",),
                "middle_block": ("GM_MIDDLE_BLOCK",),
                "output_blocks": ("GM_OUTPUT_BLOCKS",),
                "out": ("GM_OUT",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    def merge(self, model1, model2, strength, unique_id, time_embed=None, label_emb=None, input_blocks=None, middle_block=None, output_blocks=None, out=None, **kwargs):
        for config in [time_embed, label_emb, input_blocks, middle_block, output_blocks, out]:
            if config is not None:
                kwargs.update(config)

        kwargs = {k: v * strength for k, v in kwargs.items()}
        
        sb = io.StringIO()
        layers = model1.model_state_dict().keys()
        for k, v in kwargs.items():
            if any(layer.startswith(f'diffusion_model.{k}') for layer in layers):
                sb.write(f'{k}={v}\n')
        
        if len(kwargs) > 0:
            bm = ModelMergeBlocks()
            model = bm.merge(model1, model2, **kwargs)[0]
        else:
            model = model1
        
        return (model, sb.getvalue())

class GodnessMerger_RAW_Apply:
    RETURN_TYPES = ("MODEL","STRING")
    RETURN_NAMES = ("MODEL", "RATIO_VALUES")
    FUNCTION = "merge"
    CATEGORY = "SP-Nodes/model_merging"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    def merge(self, model1, model2, strength, text: str, unique_id, **kwargs):
        lines = [line.split('=') for line in text.splitlines() if line]
        for k, v in lines:
            kwargs[k] = float(v)

        kwargs = {k: v * strength for k, v in kwargs.items()}

        if len(kwargs) > 0:
            bm = ModelMergeBlocks()
            model = bm.merge(model1, model2, **kwargs)[0]
        else:
            model = model1
        
        return (model, text)
    
class Random_Model_Merge:
    RETURN_TYPES = ("MODEL","STRING")
    RETURN_NAMES = ("MODEL", "VALUES")
    FUNCTION = "merge"
    CATEGORY = "SP-Nodes/model_merging"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  
                # "preset": (list(get_presets().keys()),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "time_embed": (["default", "random", "custom"], ),
                "time_embed_custom": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label_emb": (["default", "random", "custom"], ),
                "label_emb_custom": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out": (["default", "random", "custom"], ),
                "out_custom": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    def merge(self, model1, model2, seed, strength, min, max, time_embed, time_embed_custom, label_emb, label_emb_custom, out, out_custom, unique_id, **kwargs):
        random.seed(seed)
        
        ratios_values = [random.uniform(min, max) * strength for _ in range(25)]
        block_types = ["input_blocks", "middle_block", "output_blocks"]
        num_blocks = [12, 1, 12]

        def update_custom_value(value_type, value_default, value_custom):
            if value_type == 'custom':
                return value_custom
            elif value_type == 'random':
                return random.uniform(min, max)
            else:
                return value_default

        kwargs['time_embed.'] = update_custom_value(time_embed, 1.0, time_embed_custom)
        kwargs['label_emb.'] = update_custom_value(label_emb, 1.0, label_emb_custom)
        kwargs['out.'] = update_custom_value(out, 1.0, out_custom)
        
        for block_type, num in zip(block_types, num_blocks):
            for i in range(num):
                ratio_key = "{}.{}.".format(block_type, i) if num > 1 else "{}.".format(block_type)
                kwargs[ratio_key] = ratios_values.pop(0)
               
        sb = io.StringIO()
        layers = model1.model_state_dict().keys()
        for k, v in kwargs.items():
            if any(layer.startswith(f'diffusion_model.{k}') for layer in layers):
                sb.write(f'{k}={v}\n')
                
        bm = ModelMergeBlocks()
        model = bm.merge(model1, model2, **kwargs)
        
        return (model[0],sb.getvalue())

class GodnessMerger_NoiseInjection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "operation": (["random", "gaussian"], {'default': "gaussian"}),
                "mean": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "std": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ratio": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  
                # "preset": (list(get_presets().keys()),),
                # "preset_strength": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
       
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "SP-Nodes/model_merging"
    
    def get_patched_state(self, model):
        """Uses a Comfy ModelPatcher to get the patched state dict of a model.

        Args:
            model (ModelPatcher): The model to get the patched state dict from.
            
        Returns:
            Dict[str, torch.Tensor]: The patched state dict.
        """
        if len(model.patches) > 0:
            print("Model has patches, applying them")
            model.patch_model(None, True)
            model_sd = model.model_state_dict()
            model.unpatch_model()
        else:
            model_sd = model.model_state_dict()
        
        return model_sd

    def merge(self, model, operation, mean, std, ratio, seed, unique_id, **kwargs):
        random.seed(seed)
        torch.manual_seed(seed)
        
        m = model.clone()
        # model_sd = self.get_patched_state(m) # high vram usage
        patches = model.get_key_patches("diffusion_model.")
        model_sd = {k: patches[k][0] for k in patches}

        # kp = model.get_key_patches("diffusion_model.")
        # for k in kp:
        #     # #.to(device='cpu')
        #     # v = kp[k][0].half() #.to(device='cuda') # kp[k] = tuple (Tensor,)
        #     # v_mod =  v + torch.normal(torch.zeros_like(v), v.std() * preset_strength)
        #     # # print(f'{k}: {v.std()} -> {v_mod.std()}')
        #     # m.add_patches({k: (v_mod,)}, 1.0 - ratio, ratio)
        #     weight = kp[k][0]
        #     temp_weight = weight.to(torch.float32, copy=True)
        #     # temp_weight += torch.normal(torch.zeros_like(temp_weight), temp_weight.std() * preset_strength)
        #     temp_weight += torch.normal(0, temp_weight.std() * preset_strength, size=temp_weight.size(), device=temp_weight.device)
        #     m.add_patches({k: (comfy.model_management.cast_to_device(temp_weight, weight.device, weight.dtype),)}, 1.0 - ratio, ratio)

        for k in model_sd.keys():
            w : torch.Tensor = model_sd[k]
            a = w.to(torch.float32, copy=True)

            if operation == "random":
                # Create a random mask of the same shape as the given layer.
                t_random = torch.rand(a.shape, device=a.device) - 0.5
            else:
                # Create a gaussian noise mask of the same shape as the given layer.
                t_random = torch.normal(mean, std, size=a.shape, device=a.device) - mean
                
            result_tensor = a + t_random
            del t_random

            # Merge our tensors
            strength_patch = 1.0 - ratio
            strength_model = ratio

            m.add_patches({k: (comfy.model_management.cast_to_device(result_tensor, w.device, w.dtype),)}, strength_patch, strength_model)

        return (m, )
    
    def perf(self, model, seed, preset_strength, ratio, unique_id, **kwargs):
        random.seed(seed)
        torch.manual_seed(seed)
        
        m = model.clone()
        kp = m.get_key_patches("diffusion_model.")
        
        create_w = 0.0
        create_zeros_like = 0.0
        create_std_noise = 0.0
        create_noise = 0.0
        add_op = 0.0
        add_patches = 0.0
        
        # start_time = time.time()
        # for k in kp:
        #     weight = kp[k][0]
        #     temp_weight = weight.to(torch.float32, copy=True)
        #     v_mod =  temp_weight + torch.normal(torch.zeros_like(temp_weight), temp_weight.std() * preset_strength)
        # print(f"Simple: {time.time() - start_time}s")
        
        start_time = time.time()
        for k in kp:
            weight = kp[k][0]
            
            start_time = time.time()
            temp_weight = weight.to(torch.float32, copy=True)
            create_w += time.time() - start_time
            
            start_time = time.time()
            std_noise = temp_weight.std() * preset_strength
            create_std_noise += time.time() - start_time
            
            start_time = time.time()
            zeros_like = torch.zeros_like(temp_weight)
            create_zeros_like += time.time() - start_time
            
            start_time = time.time()
            noise = torch.normal(zeros_like, std_noise)
            # noise = torch.normal(0, std_noise, size=temp_weight.size(), device=temp_weight.device)
            create_noise += time.time() - start_time
            
            start_time = time.time()
            v_mod = temp_weight.add_(noise)
            add_op += time.time() - start_time
            
            start_time = time.time()
            m.add_patches({k: (comfy.model_management.cast_to_device(v_mod, weight.device, weight.dtype),)}, 1.0 - ratio, ratio)
            add_patches += time.time() - start_time
            
        print(f"create_w: {create_w}s")
        print(f"create_zeros_like: {create_zeros_like}s")
        print(f"create_std_noise: {create_std_noise}s")
        print(f"create_noise: {create_noise}s")
        print(f"add_op: {add_op}s")
        print(f"add_patches: {add_patches}s")
        
        # start_time = time.time()
        # for k in kp:
        #     weight = kp[k][0]
        #     temp_weight = weight.to(torch.float32, copy=True)
            
        #     noise_np = np.random.normal(0, temp_weight.std().item() * preset_strength, size=temp_weight.size())
        #     noise = torch.from_numpy(noise_np).to(temp_weight.device)
        #     v_mod = temp_weight + noise
        # print(f"inplace+numpy: {time.time() - start_time}s")

        return (m, )
    
NODE_CLASS_MAPPINGS = {
    "Random_Model_Merge": Random_Model_Merge,
    "GodnessMerger_Apply": GodnessMerger_Apply,
    "GodnessMerger_RAW_Apply": GodnessMerger_RAW_Apply,
    "GodnessMerger_TimeEmbed": GodnessMerger_TimeEmbed,
    "GodnessMerger_LabelEmb": GodnessMerger_LabelEmb,
    "GodnessMerger_InputBlocks": GodnessMerger_InputBlocks,
    "GodnessMerger_MiddleBlock": GodnessMerger_MiddleBlock,
    "GodnessMerger_OutputBlocks": GodnessMerger_OutputBlocks,
    "GodnessMerger_InputBlocksExperimental": GodnessMerger_InputBlocksExperimental,
    "GodnessMerger_MiddleBlockExperimental": GodnessMerger_MiddleBlockExperimental,
    "GodnessMerger_OutputBlocksExperimental": GodnessMerger_OutputBlocksExperimental,
    "GodnessMerger_Out": GodnessMerger_Out,
    "GodnessMerger_NoiseInjection": GodnessMerger_NoiseInjection
}
