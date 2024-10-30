from .Graph import Graph

class SP_Pipe_ToBasicPipe:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "sp_pipe": ("SP_PIPE", ),
            },
        }

    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    FUNCTION = "fn"

    def fn(self, sp_pipe):
        pipe = (sp_pipe['model'], sp_pipe['clip'], sp_pipe['vae'], sp_pipe['positive'], sp_pipe['negative'])
        return pipe,

NODE_CLASS_MAPPINGS = {
    "SP_Pipe_ToBasicPipe": SP_Pipe_ToBasicPipe,
}
