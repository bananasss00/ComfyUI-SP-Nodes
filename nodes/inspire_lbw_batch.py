class FluxInspireLbwBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "step": ("INT", {"default": 1, "min": 1, "max": 37}),
                "block_strength": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "enable_double_blocks": (["true", "false"],),
            },
            "optional": {

            },
        }
        return inputs

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("block_vector","blocks_range","summary")
    OUTPUT_IS_LIST = (True,True, False)
    FUNCTION = "doit"
    # OUTPUT_NODE = True

    CATEGORY = 'SP-Nodes'

    def doit(self, step, block_strength, enable_double_blocks):
        # Init double_blocks strength
        init = '1,' + (('1,' if enable_double_blocks == 'true' else '0,') * 19)
        limit_batch = 999
        max_index = 37  # Maximum index for looping - max blocks count for Flux

        def block_on(n):
            blocks = init
            for i in range(0, max_index + 1, step):
                blocks += ','.join([str(block_strength) if i >= n and i < n + step else '0' for _ in range(max_index - i + 1 if i + step > max_index else step)])

                if i + step <= max_index:
                    blocks += ','  # Add a comma if not the last block

            return blocks

        block_vector = [block_on(i) for i in range(0, max_index + 1, step)]
        blocks_range = [f'{i} - {i + step - 1}' for i in range(0, max_index + 1, step)]

        summary = []
        for i, r in enumerate(blocks_range):
            summary.append(f'{r} => {block_vector[i]}')

        return block_vector[:limit_batch], blocks_range[:limit_batch], '\n'.join(summary)


NODE_CLASS_MAPPINGS = {
    "FluxInspireLbwBatch": FluxInspireLbwBatch,
}
