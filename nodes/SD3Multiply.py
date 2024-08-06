


def sd3_multiply(block, m, attn_qkv, attn_proj, mlp_fc1, mlp_fc2, adaLN_modulation_1):
    sd = m.model_state_dict()

    for key in sd:
        if key.endswith(f"{block}.attn.qkv.bias") or key.endswith(f"{block}.attn.qkv.weight"):
            m.add_patches({key: (None,)}, 0.0, attn_qkv)
        if key.endswith(f"{block}.attn.proj.bias") or key.endswith(f"{block}.attn.proj.weight"):
            m.add_patches({key: (None,)}, 0.0, attn_proj)
        if key.endswith(f"{block}.mlp.fc1.bias") or key.endswith(f"{block}.mlp.fc1.weight"):
            m.add_patches({key: (None,)}, 0.0, mlp_fc1)
        if key.endswith(f"{block}.mlp.fc2.bias") or key.endswith(f"{block}.mlp.fc2.weight"):
            m.add_patches({key: (None,)}, 0.0, mlp_fc2)
        if key.endswith(f"{block}.adaLN_modulation.1.bias") or key.endswith(f"{block}.adaLN_modulation.1.weight"):
            m.add_patches({key: (None,)}, 0.0, adaLN_modulation_1)

    return m

class SD3BlocksMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "b0": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b7": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b8": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b9": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b11": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b13": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b14": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b15": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b16": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b17": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b18": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b19": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b20": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b21": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b22": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "b23": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model,
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9,
        b10, b11, b12, b13, b14, b15, b16, b17, b18, b19,
        b20, b21, b22, b23
    ):
        m = model.clone()
        sd = m.model_state_dict()
        
        for i in range(24):
            for key in sd:
                if f'joint_blocks.{i}.' in key:
                    m.add_patches({key: (None,)}, 0.0, locals()['b' + str(i)])
            
        return (m, )

class SD3Multiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "context_block_attn_qkv": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "context_block_attn_proj": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "context_block_mlp_fc1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "context_block_mlp_fc2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "context_block_adaLN_modulation_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "x_block_attn_qkv": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "x_block_attn_proj": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "x_block_mlp_fc1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "x_block_mlp_fc2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "x_block_adaLN_modulation_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model, context_block_attn_qkv, context_block_attn_proj, context_block_mlp_fc1, context_block_mlp_fc2, context_block_adaLN_modulation_1,
                x_block_attn_qkv, x_block_attn_proj, x_block_mlp_fc1, x_block_mlp_fc2, x_block_adaLN_modulation_1):
        m = model.clone()
        m = sd3_multiply('context_block', m, context_block_attn_qkv, context_block_attn_proj, context_block_mlp_fc1, context_block_mlp_fc2, context_block_adaLN_modulation_1)
        m = sd3_multiply('x_block', m, x_block_attn_qkv, x_block_attn_proj, x_block_mlp_fc1, x_block_mlp_fc2, x_block_adaLN_modulation_1)
        return (m, )
        
NODE_CLASS_MAPPINGS = {
    "SD3Multiply": SD3Multiply,
    "SD3BlocksMultiply": SD3BlocksMultiply,
}