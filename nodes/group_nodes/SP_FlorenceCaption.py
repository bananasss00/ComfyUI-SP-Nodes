import folder_paths, nodes, logging, comfy
from comfy_execution.graph import DynamicPrompt

from .Graph import Graph


class SP_FlorenceCaption:
    CATEGORY = "SP-Nodes/Group Nodes"

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "image": ("IMAGE", {"rawLink": True}),
                "model": (
                    [
                        "microsoft/Florence-2-base",
                        "microsoft/Florence-2-base-ft",
                        "microsoft/Florence-2-large",
                        "microsoft/Florence-2-large-ft",
                        "HuggingFaceM4/Florence-2-DocVQA",
                        "thwri/CogFlorence-2.1-Large",
                        "thwri/CogFlorence-2.2-Large",
                        "gokaygokay/Florence-2-SD3-Captioner",
                        "gokaygokay/Florence-2-Flux-Large",
                        "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
                        "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
                    ],
                ),
                "precision": (["fp16", "bf16", "fp32"], ),
                "task": (
                    [
                        "detailed_caption",
                        "more_detailed_caption",
                    ],
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {},
        }
        return inputs

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("caption", )
    FUNCTION = "fn"

    def fn(self, image, model, precision, task, seed):
        graph = Graph()

        florence2_model = graph.DownloadAndLoadFlorence2Model(
            None, model, precision=precision, attention="sdpa"
        )
        caption = graph.Florence2Run(image, florence2_model, "", task=task, seed=seed)[
            2
        ]

        return {
            "result": (caption,),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "SP_FlorenceCaption": SP_FlorenceCaption,
}
