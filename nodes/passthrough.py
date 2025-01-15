class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class SP_Pass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {       
                "value": (AnyType('*'), ),      
                "type": ([], ),      
            },
            "optional": {
            }, 
    }

    RETURN_TYPES = (AnyType('*'), )
    FUNCTION = "passthrough"
    CATEGORY = "_test_"

    def passthrough(self, value, type):
        return (value,)
    
    @classmethod
    def VALIDATE_INPUTS(s, type):
        return True

NODE_CLASS_MAPPINGS = {
     'SP_Pass': SP_Pass,
}

# def create_passthrough_class(name, input_type, return_type):
#     class_name = f"SP_{name.capitalize()}Pass"
#     class_dict = {
#         "INPUT_TYPES": classmethod(lambda s: {"required": {}, "optional": {name: (input_type,)}}),
#         "RETURN_TYPES": (return_type,),
#         "RETURN_NAMES": (name,),
#         "FUNCTION": "fn",
#         "CATEGORY": "SP-Nodes/passthrough",
#         "fn": lambda self, **kwargs: (kwargs.get(name),)
#     }
#     return type(class_name, (), class_dict)

# model_types = [
#     ("string", "STRING"),
#     ("int", "INT"),
#     ("float", "FLOAT"),
#     ("boolean", "BOOLEAN"),
#     ("model", "MODEL"),
#     ("clip", "CLIP"),
#     ("vae", "VAE"),
#     ("image", "IMAGE"),
#     ("latent", "LATENT"),
#     ("conditioning", "CONDITIONING"),
# ]

# for name, input_type in model_types:
#     cls = create_passthrough_class(name, input_type, input_type)
#     globals()[cls.__name__] = cls
#     NODE_CLASS_MAPPINGS[cls.__name__] = cls
