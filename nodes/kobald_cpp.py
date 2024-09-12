import requests, math

API_URL = 'http://localhost:5001/api/v1'

system_prompt = '''you are now a prompt generator for stable diffusion.

rules of prompt generation:
1) I write a short description of the desired prompt, and you reply with an "improved" prompt in English with added/refined details according to the rules of writing a prompt for Stable Diffusion.
2) if a style is specified in brackets in the format "prompt description (style)", then finalize the prompt according to this style.
3) you only write a finished prompt and nothing else!
4) the main idea should be kept and should be described at the beginning of the prompt.'''

class LLMMode:
    def __init__(self, system_tag, sys_prompt, user_tag, assistant_tag):
        self._system_tag = system_tag
        self._sys_prompt = sys_prompt
        self._user_tag = user_tag
        self._assistant_tag = assistant_tag
        self.stop_sequence = [user_tag.strip(), assistant_tag.strip()]

    def memory(self, context):
        sys_prompt = f'{self._system_tag}{self._sys_prompt}\n' if self._sys_prompt else ''
        return f'{sys_prompt}{context}\n'
    
    def prompt(self, prompt):
        return f'{self._user_tag}{prompt}{self._assistant_tag}'

class Chat_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\nUser: ',
            assistant_tag='\nKoboldAI: ')
        
class Alpaca_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\n### Instruction:\n',
            assistant_tag='\n### Response:\n')
        
class Vicuna_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\nUSER: ',
            assistant_tag='\nASSISTANT: ')

class Metharme_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='<|user|>',
            assistant_tag='<|model|>')
             
class Llama2Chat_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='[INST] ',
            assistant_tag=' [/INST]')
           
class QuestionAnswer_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\nQuestion: ',
            assistant_tag='\nAnswer: ')
           
class ChatML_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|im_start|>system\n',
            sys_prompt=sys_prompt,
            user_tag='<|im_end|>\n<|im_start|>user\n',
            assistant_tag='<|im_end|>\n<|im_start|>assistant\n')

class InputOutput_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\n{{[INPUT]}}\n',
            assistant_tag='\n{{[OUTPUT]}}\n')
           
class CommandR_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>',
            sys_prompt=sys_prompt,
            user_tag='<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>',
            assistant_tag='<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>')

class Llama3Chat_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|start_header_id|>system<|end_header_id|>\n\n',
            sys_prompt=sys_prompt,
            user_tag='<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n',
            assistant_tag='<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
           
class Phi3Mini_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|system|>\n',
            sys_prompt=sys_prompt,
            user_tag='<|end|><|user|>\n',
            assistant_tag='<|end|>\n<|assistant|>')

class Gemma2_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<start_of_turn>user\n',
            sys_prompt=sys_prompt,
            user_tag='<end_of_turn>\n<start_of_turn>user\n',
            assistant_tag='<end_of_turn>\n<start_of_turn>model\n')

class Mistral_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\n[INST] ',
            assistant_tag=' [/INST]\n')


def generate_text(api_url, system_prompt, context, prompt, temperature_override=0, llm_mode='Gemma2', preset='default', max_length=200, seed=-1):
    endpoint = f'{api_url}/generate'
    headers = {
        'Content-Type': 'application/json'
    }

    if preset=='default':
        preset_dict = {"rep_pen": 1.1, "temperature": 0.66, "top_p": 1, "top_k": 0, "top_a": 0.96, "typical": 0.6, "tfs": 1, "rep_pen_range": 1024, "rep_pen_slope": 0.7, "sampler_order": [6, 4, 5, 1, 0, 2, 3]}
    elif preset=='simple_logical':
        preset_dict = {"rep_pen": 1.01, "temperature": 0.25, "top_p": 0.6, "top_k": 100, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5]}
    elif preset=='simple_balanced':
        preset_dict = {"rep_pen": 1.07, "temperature": 0.7, "top_p": 0.92, "top_k": 100, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5]}
    elif preset=='simple_creative':
        preset_dict = {"rep_pen": 1.15, "temperature": 1, "top_p": 0.98, "top_k": 100, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5]}
    elif preset=='silly_tavern':
        preset_dict = {"rep_pen": 1.18, "temperature": 0.7, "top_p": 0.6, "top_k": 40, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 1024, "rep_pen_slope": 0.8, "sampler_order": [6, 0, 1, 3, 4, 2, 5]}
    elif preset=='coherent_creativity':
        preset_dict = {"rep_pen": 1.2, "temperature": 0.5, "top_p": 1, "top_k": 0, "top_a": 0, "typical": 1, "tfs": 0.99, "rep_pen_range": 2048, "rep_pen_slope": 0, "sampler_order": [6, 5, 0, 2, 3, 1, 4]}
    elif preset=='godlike':
        preset_dict = {"rep_pen": 1.1, "temperature": 0.7, "top_p": 0.5, "top_k": 0, "top_a": 0.75, "typical": 0.19, "tfs": 0.97, "rep_pen_range": 1024, "rep_pen_slope": 0.7, "sampler_order": [6, 5, 4, 3, 2, 1, 0]}
    elif preset=='liminal_drift':
        preset_dict = {"rep_pen": 1.1, "temperature": 0.66, "top_p": 1, "top_k": 0, "top_a": 0.96, "typical": 0.6, "tfs": 1, "rep_pen_range": 1024, "rep_pen_slope": 0.7, "sampler_order": [6, 4, 5, 1, 0, 2, 3]}
    else:
        raise Exception('bad arg')

    mode: LLMMode = globals()[f'{llm_mode}_LLMMode'](system_prompt)

    payload = {
        "n": 1,
        # "max_context_length": 8192, 
        "max_length": max_length, 
        'prompt': mode.prompt(prompt) ,#f"\nUser:{prompt}\nAI:",
        'memory': mode.memory(context), #system_prompt,
        "sampler_seed": seed,
        # "dry_sequence_breakers": ["\n", ":", "\"", "*"],
        "trim_stop": True,
        "stop_sequence": mode.stop_sequence, #["User:", "\nUser ", "\nAI: "],
        "quiet": True,
        "use_default_badwordsids": False,
        "bypass_eos": False,
        "logit_bias": {},
        "presence_penalty": 0,
        "dry_allowed_length": 2,
        "dry_base": 1.75,
        "dry_multiplier": 0,
        "render_special": False,
        "banned_tokens": [],
        "smoothing_factor": 0,
        "dynatemp_exponent": 1,
        "dynatemp_range": 0,
        "min_p": 0
    }

    payload.update(preset_dict)

    if not math.isclose(temperature_override, 0, abs_tol=1e-4):
        print(f'temperature_override: {temperature_override}')
        payload["temperature"] = temperature_override
    
    response = requests.post(endpoint, json=payload, headers=headers)
    
    if response.status_code == 200:
        text = response.json()['results'][0]['text']
        return text
    else:
        return f'Error: {response.status_code} - {response.text}'
    
class SP_KoboldCpp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "api_url": ("STRING", {"default": API_URL, "multiline": False}),
                        "system_prompt": ("STRING", {"default": system_prompt, "multiline": True}),
                        "prompt": ("STRING", {"default": '', "multiline": True}),
                        "llm_mode": (['Chat', 'Alpaca', 'Vicuna', 'Metharme',
                            'Llama2Chat', 'QuestionAnswer', 'ChatML',
                            'InputOutput', 'CommandR', 'Llama3Chat', 
                            'Phi3Mini', 'Gemma2', 'Mistral'], ),
                        "preset": (['simple_logical', 'default', 'simple_balanced',
                                    'simple_creative', 'silly_tavern', 'coherent_creativity',
                                    'godlike', 'liminal_drift'], ),
                        "temperature_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                        "max_length": ("INT", {"default": 100, "min": 10, "max": 512}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    RETURN_TYPES = ('STRING',)
    FUNCTION = "fn"

    OUTPUT_NODE = False

    CATEGORY = "SP-Nodes"

    def fn(self, api_url, system_prompt, prompt, llm_mode, preset, temperature_override, max_length, seed):
        return generate_text(api_url, system_prompt, '', prompt, temperature_override, llm_mode, preset, max_length=max_length, seed=seed).replace('User:', ''),

class SP_KoboldCppWithContext:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "api_url": ("STRING", {"default": API_URL, "multiline": False}),
                        "system_prompt": ("STRING", {"default": system_prompt, "multiline": True}),
                        "context": ("STRING", {"default": '', "multiline": True}),
                        "prompt": ("STRING", {"default": '', "multiline": True}),
                        "llm_mode": (['Chat', 'Alpaca', 'Vicuna', 'Metharme',
                            'Llama2Chat', 'QuestionAnswer', 'ChatML',
                            'InputOutput', 'CommandR', 'Llama3Chat', 
                            'Phi3Mini', 'Gemma2', 'Mistral'], ),
                        "preset": (['simple_logical', 'default', 'simple_balanced',
                                    'simple_creative', 'silly_tavern', 'coherent_creativity',
                                    'godlike', 'liminal_drift'], ),
                        "temperature_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                        "max_length": ("INT", {"default": 100, "min": 10, "max": 512}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    RETURN_TYPES = ('STRING',)
    FUNCTION = "fn"

    OUTPUT_NODE = False

    CATEGORY = "SP-Nodes"

    def fn(self, api_url, system_prompt, context, prompt, llm_mode, preset, temperature_override, max_length, seed):
        return generate_text(api_url, system_prompt, context, prompt, temperature_override, llm_mode, preset, max_length=max_length, seed=seed).replace('User:', ''),

NODE_CLASS_MAPPINGS = {
    "SP_KoboldCpp": SP_KoboldCpp,
    "SP_KoboldCppWithContext": SP_KoboldCppWithContext,
}
