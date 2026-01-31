import codecs
from collections import OrderedDict
import copy
import json
import logging
import requests, math
from PIL import Image
from io import BytesIO
import base64
import numpy as np

API_URL = 'http://localhost:5001/api/v1'

system_prompt = '''Answer in English. I give you a topic and you write a short description on that topic. The descriptions should be a few sentences long.
I give you a theme, and you write a short description of the photo in a surrealistic style on that theme. Descriptions should be a few sentences long'''

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

class KoboldCppAuto_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='{{[SYSTEM]}}',
            sys_prompt=sys_prompt,
            user_tag='{{[INPUT]}}',
            assistant_tag='{{[OUTPUT]}}')
        
class DeepSeek25_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='<｜end▁of▁sentence｜><｜User｜>',
            assistant_tag='<｜end▁of▁sentence｜><｜Assistant｜>')
        
class OpenaiHarmony_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|start|>developer<|message|>',
            sys_prompt=sys_prompt,
            user_tag='<|end|><|start|>user<|message|>',
            assistant_tag='<|end|><|start|>assistant<|channel|>final<|message|>')
        
class GLM4_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|system|>\n',
            sys_prompt=sys_prompt,
            user_tag='<|user|>\n',
            assistant_tag='<|assistant|>\n')
        
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

class Llama4Chat_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|header_start|>system<|header_end|>\n\n',
            sys_prompt=sys_prompt,
            user_tag='<|eot|><|header_start|>user<|header_end|>\n\n',
            assistant_tag='<|eot|><|header_start|>assistant<|header_end|>\n\n')


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
           
class Phi3Mini_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<|system|>\n',
            sys_prompt=sys_prompt,
            user_tag='<|end|><|user|>\n',
            assistant_tag='<|end|>\n<|assistant|>')

class Gemma23_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='<start_of_turn>user\n',
            sys_prompt=sys_prompt,
            user_tag='<end_of_turn>\n<start_of_turn>user\n',
            assistant_tag='<end_of_turn>\n<start_of_turn>model\n')

class MistralNonTekken_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='',
            sys_prompt=sys_prompt,
            user_tag='\n[INST] ',
            assistant_tag=' [/INST]\n')
        
class MistralTekken_LLMMode(LLMMode):
    def __init__(self, sys_prompt):
        super().__init__(
            system_tag='[SYSTEM_PROMPT]',
            sys_prompt=sys_prompt,
            user_tag='\n[INST] ',
            assistant_tag=' [/INST]\n')

class OverrideCfg:
    def __init__(self, temperature, min_p, xtc_probability, xtc_threshold, smoothing_factor, smoothing_curve, dynatemp_range=0, dry_multiplier=0, dry_base=1.75, dry_allowed_length=2):
        self.temperature = temperature
        self.min_p = min_p
        self.xtc_probability = xtc_probability
        self.xtc_threshold = xtc_threshold
        self.smoothing_factor = smoothing_factor
        self.smoothing_curve = smoothing_curve
        self.dynatemp_range = dynatemp_range
        self.dry_multiplier = dry_multiplier
        self.dry_base = dry_base
        self.dry_allowed_length = dry_allowed_length

    def apply(self, payload):
        if not self.is_null(self.temperature):
            payload["temperature"] = self.temperature
        
        if not self.is_null(self.min_p):
            payload["min_p"] = self.min_p
        
        if not self.is_null(self.xtc_probability):
            payload["xtc_probability"] = self.xtc_probability
            payload["xtc_threshold"] = self.xtc_threshold

        if not self.is_null(self.smoothing_factor):
            payload["smoothing_factor"] = self.smoothing_factor
            payload["smoothing_curve"] = self.smoothing_curve

        if not self.is_null(self.dynatemp_range):
            payload["dynatemp_range"] = self.dynatemp_range

        if not self.is_null(self.dry_multiplier):
            payload["dry_multiplier"] = self.dry_multiplier
            payload["dry_base"] = self.dry_base
            payload["dry_allowed_length"] = self.dry_allowed_length

    def is_null(self, value):
        return math.isclose(value, 0, abs_tol=1e-4)

def generate_text(api_url, system_prompt, context, prompt, override_cfg: OverrideCfg = None, banned_tokens: list[str] = None, images = None, llm_mode='Gemma2', preset='default', max_length=200, seed=-1):
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
        'prompt': mode.prompt(prompt) ,#f"\nUser:{prompt}\nAI:",
        'memory': mode.memory(context), #system_prompt,
        "sampler_seed": seed,
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
        "dry_penalty_last_n": 360,
        "dry_sequence_breakers": ["\n", ":", "\"", "*"],
        "render_special": False,
        "banned_tokens": [],
        "smoothing_factor": 0,
        "dynatemp_exponent": 1,
        "dynatemp_range": 0,
        "min_p": 0,
        "xtc_probability": 0,
        "xtc_threshold": 0.5
    }

    payload.update(preset_dict)

    if override_cfg is not None:
        override_cfg.apply(payload=payload)
    
    if max_length:
        payload['max_length'] = max_length
        
    if banned_tokens is not None:
        payload['banned_tokens'] = banned_tokens
    
    if images is not None:
        images_b64 = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = base64.b64encode(buffered.getvalue())
            images_b64.append(str(img_bytes, 'utf-8'))
        payload['images'] = images_b64
    
    response = requests.post(endpoint, json=payload, headers=headers)
    
    preset_cfg = json.dumps(payload, indent=4)

    if response.status_code == 200:
        text = response.json()['results'][0]['text']
        return text, preset_cfg
    else:
        return f'Error: {response.status_code} - {response.text}', preset_cfg

class SP_KoboldCpp_OverrideCfg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "temperature": (
                            "FLOAT", {
                                "default": 0.0,
                                "min": 0.0,
                                "max": 10.0,
                                "step": 0.05,
                                "tooltip": (
                                    "Controls the randomness of predictions. Higher values make the output more creative and unpredictable, lower values make it more focused and deterministic."
                                    "\n\n"
                                    "Управляет случайностью генерации. Более высокие значения делают текст креативнее и менее предсказуемым, низкие — более логичным и последовательным."
                                )
                            }
                        ),
                        "dynatemp_range": (
                            "FLOAT", {
                                "default": 0.0,
                                "min": 0.0,
                                "max": 10.0,
                                "step": 0.1,
                                "tooltip": (
                                    "Dynamically adjusts temperature based on model uncertainty. If the model is unsure, temperature increases for more creativity; if confident, temperature decreases for accuracy. 0 disables this feature."
                                    "\n\n"
                                    "Динамически изменяет температуру в зависимости от уверенности модели. При неуверенности температура повышается для креативности, при уверенности — понижается для точности. 0 отключает функцию."
                                )
                            }
                        ),
                        "min_p": (
                            "FLOAT", {
                                "default": 0.0,
                                "min": 0.0,
                                "max": 1.0,
                                "step": 0.01,
                                "tooltip": (
                                    "Filters out tokens until their cumulative probability reaches the min_p threshold. Helps prevent repetition and improves generation quality. 0 disables this sampler."
                                    "\n\n"
                                    "Отсекает токены, пока их суммарная вероятность не достигнет порога min_p. Помогает избежать повторов и повысить качество генерации. 0 отключает семплер."
                                )
                            }
                        ),
                        "xtc_probability": (
                            "FLOAT", {
                                "default": 0.0,
                                "min": 0.0,
                                "max": 1.0,
                                "step": 0.01,
                                "tooltip": (
                                    "Probability of activating the XTC (Exclude Top Choices) sampler for each token. XTC removes the most likely words, forcing the model to be more creative."
                                    "\n\n"
                                    "Вероятность активации семплера XTC (исключение самых вероятных токенов) для каждого токена. XTC заставляет модель выбирать менее очевидные слова."
                                )
                            }
                        ),
                        "xtc_threshold": (
                            "FLOAT", {
                                "default": 0.5,
                                "min": 0.0,
                                "max": 1.0,
                                "step": 0.01,
                                "tooltip": (
                                    "Minimum probability a token must have to be removed by XTC. All candidates above this threshold are excluded except the least likely one."
                                    "\n\n"
                                    "Минимальная вероятность токена для удаления через XTC. Все кандидаты выше этого порога исключаются, кроме самого маловероятного."
                                )
                            }
                        ),
                        "smoothing_factor": (
                            "FLOAT", {
                                "default": 0,
                                "min": 0.0,
                                "max": 10.0,
                                "step": 0.01,
                                "tooltip": (
                                    "Strength of logit smoothing. Smooths the probability distribution, reducing the gap between likely and unlikely tokens. 0 disables smoothing."
                                    "\n\n"
                                    "Сила сглаживания логитов. Сглаживает распределение вероятностей, уменьшая разницу между вероятными и маловероятными токенами. 0 отключает сглаживание."
                                )
                            }
                        ),
                        "smoothing_curve": (
                            "FLOAT", {
                                "default": 1.0,
                                "min": 0.0,
                                "max": 10.0,
                                "step": 0.01,
                                "tooltip": (
                                    "Shape of the smoothing curve used in logit smoothing. Adjust for different smoothing behaviors."
                                    "\n\n"
                                    "Форма кривой для сглаживания логитов. Меняйте для разных вариантов сглаживания."
                                )
                            }
                        ),
                        # "dry_multiplier": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step": 0.01}),
                        # "dry_base": ("FLOAT", {"default": 1.75, "min": 0.0, "max": 8, "step": 0.01}),
                        # "dry_allowed_length": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                    },
                
                }

    RETURN_TYPES = ('OVERRIDE_CFG',)
    FUNCTION = "fn"

    OUTPUT_NODE = False

    CATEGORY = "SP-Nodes"
    DESCRIPTION = "Override settings like temperature, min_p, xtc, smoothing, and dynatemp. Set to 0 to use default value."

    def fn(self, temperature, dynatemp_range, min_p, xtc_probability, xtc_threshold, smoothing_factor, smoothing_curve):
        return OverrideCfg(temperature=temperature, dynatemp_range=dynatemp_range, min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold, smoothing_factor=smoothing_factor, smoothing_curve=smoothing_curve),

class SP_KoboldCpp_BannedTokens:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "delimiter": ("STRING", {"default": "\\n", "multiline": False}),
                        "tokens": ("STRING", {"default": "banned_token\nbanned phrase\nbanned phrase\\n with newline", "multiline": True}),
                    },
                
                }

    RETURN_TYPES = ('BANNED_TOKENS',)
    FUNCTION = "fn"

    OUTPUT_NODE = False

    CATEGORY = "SP-Nodes"

    def fn(self, delimiter, tokens):
        delimiter=codecs.decode(delimiter, 'unicode_escape')
        
        # Remove carriage return characters
        cleaned_tokens = tokens.replace('\r', '')

        # Split the string into a list of lines
        lines = cleaned_tokens.split(delimiter)

        # Replace escaped newline characters with actual newline characters
        processed_lines = [line.replace('\\n', '\n') for line in lines]
        return processed_lines,
    

class SP_KoboldCpp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "api_url": ("STRING", {"default": API_URL, "multiline": False}),
                        "system_prompt": ("STRING", {"default": system_prompt, "multiline": True}),
                        "prompt": ("STRING", {"default": '', "multiline": True}),
                        "llm_mode": (['KoboldCppAuto', 'Chat', 'Alpaca', 'ChatML', 'CommandR', 'DeepSeek25',
                                      'Gemma23', 'GLM4', 'Llama2Chat', 'Llama3Chat', 'Llama4Chat', 'Metharme',
                                      'MistralNonTekken', 'MistralTekken', 'Phi3Mini', 'Vicuna', 'OpenaiHarmony',
                            'QuestionAnswer',
                            'InputOutput', 
                            ], ),
                        "preset": (['simple_logical', 'default', 'simple_balanced',
                                    'simple_creative', 'silly_tavern', 'coherent_creativity',
                                    'godlike', 'liminal_drift'], ),
                        "max_length": ("INT", {"default": 0, "min": 0, "max": 8192}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                    "optional": {
                        "override_cfg": ("OVERRIDE_CFG", ),
                        "banned_tokens": ("BANNED_TOKENS", ),
                        "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response."},),
                    }
                }

    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('text','payload')
    FUNCTION = "fn"

    OUTPUT_NODE = False

    CATEGORY = "SP-Nodes"

    def fn(self, api_url, system_prompt, prompt, llm_mode, preset, max_length, seed, context='', override_cfg=None, banned_tokens=None, images=None):
        text, payload = generate_text(api_url, system_prompt, context, prompt, override_cfg, banned_tokens, images, llm_mode, preset, max_length=max_length, seed=seed)
        return text.replace('User:', ''), payload

class SP_KoboldCppWithContext(SP_KoboldCpp):
    @classmethod
    def INPUT_TYPES(s):
        input_types = dict(super().INPUT_TYPES())

        required = input_types["required"]
        
        new_required = {}
        for key, value in required.items():
            new_required[key] = value

            if key == "system_prompt":
                new_required["context"] = ("STRING", {"default": '', "multiline": True})

        input_types["required"] = new_required

        return input_types

NODE_CLASS_MAPPINGS = {
    "SP_KoboldCpp": SP_KoboldCpp,
    "SP_KoboldCppWithContext": SP_KoboldCppWithContext,
    "SP_KoboldCpp_OverrideCfg": SP_KoboldCpp_OverrideCfg,
    "SP_KoboldCpp_BannedTokens": SP_KoboldCpp_BannedTokens,
}
