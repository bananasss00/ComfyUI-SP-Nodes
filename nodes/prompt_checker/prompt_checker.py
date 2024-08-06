from .wildcards import process, get_wildcard_list
from server import PromptServer
from aiohttp import web

DISABLED_TOKEN = '🔒'
CATEGORY = "SP-Nodes"

@PromptServer.instance.routes.get("/prompt_checker/wildcards/list")
async def wildcards_list(request):
    data = {'data': get_wildcard_list()}
    return web.json_response(data)

class PromptChecker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": 'ur prompt', 'multiline': True, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
            }
        }

    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('prompt','wildcard')
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(s, prompt, seed, **kwargs):
        tokens = [t for t in s.tokenize(prompt) if not t.startswith(DISABLED_TOKEN)]
        prompt = ', '.join(tokens)
        wildcard_prompt = process(prompt, seed)
        return prompt, wildcard_prompt,

    def tokenize(self, s):
        tokens = []
        current_token = ''
        inside_brackets = 0
        inside_parentheses = False

        for char in s:
            if char == '{':
                inside_brackets += 1
                current_token += char
            elif char == '}':
                inside_brackets -= 1
                current_token += char
            elif char == '(':
                inside_parentheses = True
                current_token += char
            elif char == ')':
                inside_parentheses = False
                current_token += char
            elif char == ',' and inside_brackets == 0 and not inside_parentheses:
                tokens.append(current_token.strip())
                current_token = ''
            else:
                current_token += char

        if current_token:
            tokens.append(current_token.strip())

        return [token for token in tokens if token]

NODE_CLASS_MAPPINGS = {
    "PromptChecker": PromptChecker, 
}