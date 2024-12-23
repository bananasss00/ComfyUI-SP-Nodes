
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
    
class SP_DynamicCombo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected": ("STRING", {'default': ''}),
            },
        }

    RETURN_TYPES = (AnyType('*'),)

    FUNCTION = "main"
    CATEGORY = 'SP-Nodes'
    DESCRIPTION = 'Configure combo values in node properties'

    def main(self, selected):
        return (selected,)
    
NODE_CLASS_MAPPINGS = {
    "SP_DynamicCombo": SP_DynamicCombo,
}