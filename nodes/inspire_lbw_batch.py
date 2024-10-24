import math

def round_if_close(num, tol=1e-5):
    nearest_int = round(num)
    
    if abs(num - nearest_int) <= tol:
        return nearest_int
    else:
        return num
    
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

class FluxInspireLbw_Batch:
    DOUBLE_BLOCKS = 19
    SINGLE_BLOCKS = 37

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "block_strength": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "double_blocks": ('BOOLEAN', {"default": False}),
                "single_blocks": ('BOOLEAN', {"default": False}),
                "force_double_blocks": ("STRING", {"default": '', 'multiline': True, "dynamicPrompts": False, "tooltip": "example: 1,5,6,8"}),
                "force_single_blocks": ("STRING", {"default": '', 'multiline': True, "dynamicPrompts": False, "tooltip": "example: 1,5,6,8"}),
                "forced_block_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "reverse_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {

            },
        }
        return inputs

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("block_vector","blocks_range","summary")
    OUTPUT_IS_LIST = (True,True, False)
    FUNCTION = "process_blocks"
    # OUTPUT_NODE = True

    CATEGORY = 'SP-Nodes'

    def process_blocks(self, block_strength, double_blocks, single_blocks, force_double_blocks, force_single_blocks, forced_block_strength, reverse_mode) -> str:
        block_strength = round_if_close(block_strength, tol=1e-5)
        forced_block_strength = round_if_close(forced_block_strength, tol=1e-5)

        block_vector_batch = []
        block_num_batch = []
        summary = []

        force_double_blocks = force_double_blocks.split(',')
        force_single_blocks = force_single_blocks.split(',')

        def is_forced(i):
            return self.is_double_block(i) and str(i) in force_double_blocks or self.is_single_block(i) and str(i - self.DOUBLE_BLOCKS) in force_single_blocks

        def block_on(n):
            block_vector = ['1']
            for i in range(0, self.blocks_count() + 1):
                if is_forced(i):
                    block_vector.append(str(forced_block_strength) if not reverse_mode else '0')
                else:
                    # if not reverse_mode:
                    #     value = str(block_strength) if i == n else '0'
                    # else:
                    #     value = str(block_strength) if i != n else '0'
                    value = str(block_strength if (i != n) ^ (not reverse_mode) else '0')
                    block_vector.append(value)
                    
            return ','.join(block_vector)
        
        for i in range(0, self.blocks_count() + 1):
            if self.is_double_block(i) and not double_blocks:
                continue
            if self.is_single_block(i) and not single_blocks:
                continue
            if is_forced(i):
                continue
            
            block_vector = block_on(i)
            block_vector_batch.append(block_vector)
            
            num = f'd.{i}' if self.is_double_block(i) else f's.{i - self.DOUBLE_BLOCKS}' 
            block_num_batch.append(num)

        for i, preset in enumerate(block_vector_batch):
            summary.append(block_num_batch[i] + f' => {preset}')

        return block_vector_batch, block_num_batch, '\n'.join(summary)
            
    def blocks_count(self):
        return self.DOUBLE_BLOCKS + self.SINGLE_BLOCKS
    
    def is_double_block(self, i):
        return i < self.DOUBLE_BLOCKS

    def is_single_block(self, i):
        return i >= self.DOUBLE_BLOCKS

class FluxInspireLbw_BlockVectorPreset:
    DOUBLE_BLOCKS = 19
    SINGLE_BLOCKS = 37

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "enable_double_blocks": ("STRING", {"default": '', 'multiline': True, "dynamicPrompts": False, "tooltip": "example: 1,5,6,8"}),
                "enable_single_blocks": ("STRING", {"default": '', 'multiline': True, "dynamicPrompts": False, "tooltip": "example: 1,5,6,8"}),
                "reverse_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {

            },
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("block_vector",)
    FUNCTION = "process"
    # OUTPUT_NODE = True

    CATEGORY = 'SP-Nodes'

    def process(self, enable_double_blocks, enable_single_blocks, reverse_mode):
        enable_double_blocks = [int(b.strip()) for b in enable_double_blocks.strip().split(',')] if enable_double_blocks else []
        enable_single_blocks = [int(b.strip()) for b in enable_single_blocks.split(',')] if enable_single_blocks else []

        if len(enable_double_blocks) == 0 and len(enable_single_blocks) == 0:
            return ('1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1', )

        preset = ['1']
        for i in range(self.DOUBLE_BLOCKS + self.SINGLE_BLOCKS + 1):
            if i < self.DOUBLE_BLOCKS and i in enable_double_blocks or \
               i >= self.DOUBLE_BLOCKS and i - self.DOUBLE_BLOCKS in enable_single_blocks:
                preset.append('1' if not reverse_mode else '0')
            else:
                preset.append('0' if not reverse_mode else '1')

        return (','.join(preset), )    



NODE_CLASS_MAPPINGS = {
    "FluxInspireLbw_Batch": FluxInspireLbw_Batch,
    "FluxInspireLbw_BlockVectorPreset": FluxInspireLbw_BlockVectorPreset,
}
