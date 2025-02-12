import json
import os
import re
import comfy.ldm.modules.attention
import numpy as np
import folder_paths
from PIL import Image, ImageOps
import io
import torch
import requests
from collections import OrderedDict
from PIL.PngImagePlugin import PngInfo
from collections import namedtuple
from datetime import datetime, timedelta
import sys
import random
import nltk
import contextlib
import codecs
import logging
import itertools
import importlib

import comfy.model_management as mm
from comfy.cli_args import args
from comfy.comfy_types import IO

import comfy, comfy_extras
from comfy_extras.nodes_tomesd import TomePatchModel
import comfy_extras.nodes_freelunch as nodes_freelunch

from aiohttp import web
from folder_paths import recursive_search, filter_files_extensions
import comfy.ldm
import comfy.ldm.modules
from server import PromptServer

# ANSI escape codes for colors
RED = '\033[91m'
BLUE = '\033[94m'
GRAY = '\033[90m'
RESET = '\033[0m'  # Reset color to default

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# with open('a:/modules.txt', 'w') as f:
#     f.write('\n'.join(sys.modules.keys()))

def dump_mods():
    with open('modules.txt', 'w', encoding='utf-8') as f:
        f.write(str(sys.modules))

# Define module constants after checking for their presence
ANY_TYPE = AnyType("*")
NODES = sys.modules['nodes']

API_URL = f"https://api.telegram.org/bot"

CATEGORY = "SP-Nodes"


class ImgMetaValueExtractor:
    # batch i2i generated images(upscale/anything)
    def __init__(s):
        s.index = 0

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"path": ("STRING", {"default": '', "multiline": False}),
            "prompt_type": (["prompt", "workflow"],),
            "value1": ("STRING", {"default": "[9][inputs][seed]"}), "value2": ("STRING", {"default": ""}),
            "value3": ("STRING", {"default": ""}), "value4": ("STRING", {"default": ""}),
            "value5": ("STRING", {"default": ""}), # "input": (ANY_TYPE, ),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, "STRING")
    RETURN_NAMES = ("image", "file_name_no_ext", "out1", "out2", "out3", "out4", "out5", "png_info")
    FUNCTION = "doit"
    # OUTPUT_NODE = True

    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def doit(s, path, prompt_type, value1, value2, value3, value4, value5):
        png_path = s._get_next_png(path)
        print(f'[{s.index}] png_path: {png_path}')
        img = Image.open(png_path)
        image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        img.close()

        file_extension = os.path.splitext(png_path)[1].lower()

        try:
            if file_extension == '.png' and isinstance(img.info, dict) and prompt_type in img.info:
                info = img.info
            elif file_extension == '.webp':
                info = s._extract_metadata_from_webp(png_path)
            else:
                return (image, None, None, None, None, None, None, None,)
            
            workflow = json.loads(info[prompt_type])
    
            return (image,
                    os.path.splitext(os.path.basename(png_path))[0],
                    s._read_value(workflow, value1, prompt_type),
                    s._read_value(workflow, value2, prompt_type),
                    s._read_value(workflow, value3, prompt_type),
                    s._read_value(workflow, value4, prompt_type),
                    s._read_value(workflow, value5, prompt_type),
                    s._comfyui_prompt_to_str(info)
                    )
        except Exception as ex:
            # raise ex
            pass

        return (image, None, None, None, None, None, None, None,)

    def _extract_metadata_from_webp(self, file_path):
        img = Image.open(file_path)

        metadata = img.getexif()

        if metadata:
            prompt_workflow_keys = [0x0110, 0x010f]
            result = {}
            for key in prompt_workflow_keys:
                info = metadata[key]
                k, v = info.split(':', 1)
                result[k] = v
            return result
        else:
            return None
    
    def _comfyui_prompt_to_str(s, info):
        result = {}
        for key in ['prompt', 'workflow']:
            if key not in info:
                continue
            result[key] = json.loads(info[key])
        return json.dumps(result, indent=4)
    
    def _read_value(s, workflow, path, prompt_type):
        # todo: better parser with exception handler and print readable reason!
        if not path:
            return None

        value = None

        try:
            matches = re.findall(r'\[(.*?)\]', path)
            print(f'path: {path}, matches: {matches}')
            value = workflow['nodes'] if prompt_type == 'workflow' else workflow
            for i, m in enumerate(matches):
                if prompt_type == 'workflow' and i == 0:
                    value = next((v for v in value if v['id'] == int(m)), None)
                    print(f'value: {value}')
                else:
                    m_new = int(m) if isinstance(value, list) else m
                    print(f'm: {m}, m_new: {m_new}, m_new_type: {type(m_new)}')
                    value = value[m_new]
        except Exception as ex:
            print(f"Can't parse path {path} in workflow")
            # raise ex

        return value

    def _find_pngs(s, directory):
        files = []
        for root, dirs, subs in os.walk(directory):
            for file in subs:
                if not file.endswith('.png') and not file.endswith('.webp'):
                    continue
                full_path = os.path.join(root, file)
                files.append(full_path)
        files.sort()
        return files

    def _get_next_png(s, path):
        files = s._find_pngs(path)

        if len(files) == 0:
            return None

        if s.index >= len(files):
            s.index = 0

        file = files[s.index]
        s.index += 1
        return file


class SendTelegramChatBot:
    def __init__(self):
        self._album = list()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",), "bot_token_env": ("STRING", {"default": "TG_BOT_TOKEN"}), "chat_id_env": ("STRING", {"default": "TG_BOT_CHATID"}),
                "compress": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}),
                "send_as_document": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                # "include_prompt": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "album_size": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = CATEGORY

    def _send_media_group(self, token, chat_id, images, caption='', compress=False, send_as_document=False):
        files = {}
        media = []
        for i, img in enumerate(images):
            with io.BytesIO() as output:
                if compress:
                    img[0].save(output, 'JPEG', quality=80)
                else:
                    img[0].save(output, 'PNG', pnginfo=img[1], compress_level=4)
                output.seek(0)
                ext = 'jpg' if compress else 'png'
                name = f'photo{i}.{ext}'
                files[name] = output.read()
                media.append(dict(type='document' if send_as_document else 'photo', media=f'attach://{name}'))

        media[0]['caption'] = caption
        response = requests.post(f'{API_URL}{token}/sendMediaGroup',
                                 data={'chat_id': chat_id, 'media': json.dumps(media), 'parse_mode': None}, files=files)

        if response.status_code == 200:
            print("Image sent successfully!")
        else:
            print("Error sending image:", response.reason)

        return response

    def doit(self, images, bot_token_env, chat_id_env, compress, send_as_document, album_size, prompt=None,
             extra_pnginfo=None):
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            self._album.append((img, metadata))

            if len(self._album) >= album_size:
                self._send_media_group(os.getenv(bot_token_env), chat_id=os.getenv(chat_id_env), images=self._album,
                                       caption='', compress=compress,
                                       send_as_document=send_as_document)
                self._album.clear()

        return (None,)

    def convert_png_to_jpeg(self, img):
        img = img.convert('RGB')
        bio = io.BytesIO()
        img.save(bio, 'JPEG', quality=80)
        return Image.open(bio)  # return bio.getvalue()


class BoolSwitchOutStr:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "enabled": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                    }
                }

    # RETURN_TYPES = (any_typ, )
    # RETURN_NAMES = ("output", )
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("state", )
    FUNCTION = "doit"
    # OUTPUT_NODE = True

    CATEGORY = CATEGORY

    def doit(s, enabled):
        print(f'enabled: {enabled}')
        # return (input, ) if enabled else None
        return ('True',) if enabled else ('False',)


class LoraLoaderByPath:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_path": ("STRING", {"default": "c:\\loras\\my_lora.safetensors"}),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = CATEGORY

    def load_lora(self, model, clip, lora_path, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class LoraLoaderOnlyModelByPath:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_path": ("STRING", {"default": "c:\\loras\\my_lora.safetensors"}),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"

    CATEGORY = CATEGORY

    def load_lora(self, model, lora_path, strength_model):
        if strength_model == 0:
            return (model)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0.0)
        return (model_lora,)

@PromptServer.instance.routes.post("/sp_nodes/lora_from_folder_list")
async def lora_from_folder_list(request):
    try:
        data = await request.json()

        lora_folder = data['lora_folder']
        
        files, folders_all = recursive_search(lora_folder, excluded_dir_names=['.git'])
        files = filter_files_extensions(files, ['.safetensors'])
        # logging.info(f'lora_folder: {lora_folder}, files: {files}')
        

        return web.json_response({'data': files})
    finally:
        pass
    
    return web.json_response({'data': []}) 

class LoraLoaderFromFolder:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_folder": ("STRING", {"default": r""}),
                "lora_name": ([""],),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "clip_opt": ("CLIP", ),
            }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = CATEGORY

    def load_lora(self, model, lora_folder, lora_name, strength_model, strength_clip, clip_opt=None):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip_opt)

        lora_path = os.path.join(lora_folder, lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        clip = clip_opt
        if abs(strength_clip) < 1e-5:
            clip = None

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        if abs(strength_clip) < 1e-5 and clip_opt:
            clip_lora = clip_opt
            
        return (model_lora, clip_lora)

    @classmethod
    def VALIDATE_INPUTS(s, lora_name):
        return True

class RandomPromptFromBook:
    def __init__(self) -> None:
        self._sentences: list[str] = None
        self._latest_txt: str = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"book_txt": ("STRING", {"default": "c:\\books\\my_book.txt"}),
                     "min_choices": ("INT", {"default": 1, "min": 1, "max": 20}),
                     "max_choices": ("INT", {"default": 1, "min": 1, "max": 20}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "break_choises": ("BOOLEAN", {"default": True}),
                     }
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_prompt"

    CATEGORY = CATEGORY

    def get_prompt(self, book_txt, min_choices, max_choices, seed, break_choises):
        random.seed(seed)

        if not self._sentences or self._latest_txt != book_txt:
            with open(book_txt, 'r', encoding='utf-8') as f:
                text = f.read().replace('\t', ' ').replace('\r\n', '\n')
                text = self._clean_string(text)
                sentences = [self._replace_last_punctuation(sentence.replace("\n", " ")) for sentence in nltk.sent_tokenize(text)]
                self._sentences = sentences
                self._latest_txt = book_txt

        choices = min_choices
        if min_choices < max_choices:
            choices = random.randint(min_choices, max_choices)

        sep = ' BREAK ' if break_choises else ', '
        prompt = sep.join(random.choices(self._sentences, k=choices))
        return (prompt,)
    
    def _clean_string(self, input_string):
        return re.sub("[^a-zA-Z ,.!?]", '', input_string)
    
    def _replace_last_punctuation(self, s):
        if s[-1] in ['.', ',', '!', '?']:
            s = s[:-1]
        return s

class TextSplitJoinByDelimiter:
    # from mixlab.ChatGPT.py
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,"dynamicPrompts": False}),
                "split_delimiter":("STRING", {"multiline": False,"default":",","dynamicPrompts": False}),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 1000, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                 "skip_every": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 10, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "max_count": ("INT", {
                    "default": 10,
                    "min": 1, #Minimum value
                    "max": 1000, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "join_delimiter":("STRING", {"multiline": False,"default":",","dynamicPrompts": False}),
            }
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("list_str","joined_str")
    FUNCTION = "run"
    # OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,False)

    CATEGORY = CATEGORY

    def run(self, text,split_delimiter,join_delimiter,start_index,skip_every,max_count):
         
        if split_delimiter=="":
            arr=[text.strip()]
        else:
            split_delimiter=codecs.decode(split_delimiter, 'unicode_escape')
            arr= [line for line in text.split(split_delimiter) if line.strip()]

        arr= arr[start_index:start_index + max_count * (skip_every+1):(skip_every+1)]

        join_delimiter = codecs.decode(join_delimiter, 'unicode_escape')

        return (arr,join_delimiter.join(arr),)

# from easyUse
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class StrToCombo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": ''}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ('COMBO',)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    CATEGORY = CATEGORY

    def doit(s, value):
        if not isinstance(value, list):
            value = [value]

        return value,

class ComfyuiRuntimeArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fp8_mode": (["off", "fp8_e4m3fn", "fp8_e5m2"],),
                "attention": (["sage", "sdp", "xformers"],),
                "fast": ('BOOLEAN', {"default": False}),
                "extra_reserved_vram_mb": ("INT", {"default": 600, "min": 400, "max": 24576, "step": 1}),
                "disable_smart_memory": ('BOOLEAN', {"default": False}),
            },
            "optional": {
                "opt_in":(ANY_TYPE, ),
            }
        }

    RETURN_TYPES = (ANY_TYPE, 'STRING')
    RETURN_NAMES = ('out', 'args_str')
    FUNCTION = "doit"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def doit(s, fp8_mode, attention, fast, extra_reserved_vram_mb, disable_smart_memory, opt_in=None):
        if fp8_mode == 'fp8_e4m3fn':
            args.fp8_e4m3fn_unet = args.fp8_e4m3fn_text_enc = True
            args.fp8_e5m2_unet = args.fp8_e5m2_text_enc = False
        elif fp8_mode == 'fp8_e5m2':
            args.fp8_e4m3fn_unet = args.fp8_e4m3fn_text_enc = False
            args.fp8_e5m2_unet = args.fp8_e5m2_text_enc = True
        else:
            args.fp8_e4m3fn_unet = args.fp8_e4m3fn_text_enc = False
            args.fp8_e5m2_unet = args.fp8_e5m2_text_enc = False

        attn = comfy.ldm.modules.attention
        match attention:
            case 'sage':
                comfy.model_management.XFORMERS_IS_AVAILABLE = False
                comfy.model_management.ENABLE_PYTORCH_ATTENTION = True
                args.use_sage_attention = True
                attn.optimized_attention = attn.optimized_attention_masked = attn.attention_sage
                if not hasattr(attn, 'sageattn') or attn.sageattn is None:
                    attn.sageattn = importlib.import_module('sageattention').sageattn
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            case 'sdp':
                comfy.model_management.XFORMERS_IS_AVAILABLE = False
                comfy.model_management.ENABLE_PYTORCH_ATTENTION = True
                args.use_sage_attention = False
                attn.optimized_attention = attn.optimized_attention_masked = attn.attention_pytorch
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            case 'xformers':
                comfy.model_management.XFORMERS_IS_AVAILABLE = True
                comfy.model_management.ENABLE_PYTORCH_ATTENTION = False
                args.use_sage_attention = False
                attn.optimized_attention = attn.optimized_attention_masked = attn.attention_xformers
                if not hasattr(attn, 'xformers') or attn.xformers is None:
                    attn.xformers = importlib.import_module('xformers')
                    attn.xformers.ops = importlib.import_module('xformers.ops')
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            case _:
                comfy.model_management.XFORMERS_IS_AVAILABLE = False
                comfy.model_management.ENABLE_PYTORCH_ATTENTION = False
                args.use_sage_attention = False
                attn.optimized_attention = attn.optimized_attention_masked = attn.attention_sub_quad

        args.fast = fast

        mm.EXTRA_RESERVED_VRAM = extra_reserved_vram_mb * 1024**2

        mm.DISABLE_SMART_MEMORY = disable_smart_memory

        return opt_in, str('\n'.join(f'{a[0]}={a[1]}' for a in args._get_kwargs()))


class SP_KSamplerSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("sampler_name", "scheduler")
    FUNCTION = "fn"

    CATEGORY = "SP-Nodes"

    def fn(self, sampler_name, scheduler):
        return sampler_name, scheduler


class SP_XYGrid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "X": (ANY_TYPE,),
                "Y": (ANY_TYPE,),
                # "swap_axis": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("X_LIST", "Y_LIST")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,True)
    FUNCTION = "fn"

    CATEGORY = "SP-Nodes"

    def fn(self, X, Y, swap_axis=False):
        # all_lists = [X, Y] if swap_axis[0] else [Y, X]
        all_lists = [X, Y]
        combinations = list(itertools.product(*all_lists))
        return [c[0] for c in combinations], [c[1] for c in combinations]
    
class SP_XYValues:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "value1\nvalue2\nvalue3",
                    },
                ),
                "type": (['int', 'float', 'bool', 'string'],),
                "delimiter": (
                    "STRING",
                    {
                        "multiline": False,
                        "dynamicPrompts": False,
                        "default": "\\n",
                    },
                ),
            }
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("values",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "fn"

    CATEGORY = "SP-Nodes"

    def fn(self, values, type, delimiter):
        delimiter=codecs.decode(delimiter, 'unicode_escape')
        values = values.replace('\r', '').split(delimiter)

        match type:
            case 'int':
                values = [int(v) for v in values]
            case 'float':
                values = [float(v) for v in values]
            case 'bool':
                values = [v.lower() in ['true', 'yes', '+'] for v in values]
            case _:
                pass
            
        return values,

class SP_ListAny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "value1": (ANY_TYPE,),
                "value2": (ANY_TYPE,),
                "value3": (ANY_TYPE,),
                "value4": (ANY_TYPE,),
                "value5": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("values",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "fn"

    CATEGORY = "SP-Nodes"

    def fn(self, **kwargs):
        result = []
        for value in kwargs.values():
            if value is None:
                continue
            result.extend(value)

            # if isinstance(value, list):
            #     result.extend(value)
            # else:
            #     result.append(value)
        return result,

class SP_SwitchBooleanAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": (IO.ANY, {"lazy": True}),
                "on_false": (IO.ANY, {"lazy": True}),
                "boolean": (IO.BOOLEAN, {"default": True}),
            }
        }

    CATEGORY = CATEGORY
    RETURN_TYPES = (IO.ANY,)

    FUNCTION = "execute"

    def check_lazy_status(self, *args, **kwargs):
        selected = kwargs['boolean']
        input_name = 'on_true' if selected else 'on_false'

        print(f"SELECTED: {input_name}")

        if input_name in kwargs:
            return [input_name]
        else:
            return []

    def execute(self, on_true, on_false, boolean=True):
        if boolean:
            return (on_true,)
        else:
            return (on_false,)
        

NODE_CLASS_MAPPINGS = {
    "BoolSwitchOutStr": BoolSwitchOutStr,
    "ImgMetaValueExtractor": ImgMetaValueExtractor,
    "SendTelegramChatBot": SendTelegramChatBot,
    "LoraLoaderByPath": LoraLoaderByPath,
    "LoraLoaderOnlyModelByPath": LoraLoaderOnlyModelByPath,
    "LoraLoaderFromFolder": LoraLoaderFromFolder,
    "RandomPromptFromBook": RandomPromptFromBook,
    "TextSplitJoinByDelimiter": TextSplitJoinByDelimiter,
    "StrToCombo": StrToCombo, 
    "ComfyuiRuntimeArgs": ComfyuiRuntimeArgs, 
    "SP_KSamplerSelect": SP_KSamplerSelect, 
    "SP_XYGrid": SP_XYGrid, 
    "SP_XYValues": SP_XYValues, 
    "SP_ListAny": SP_ListAny, 
    "SP_SwitchBooleanAny": SP_SwitchBooleanAny, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoolSwitchOutStr": "Bool Switch Out Str",
    "ImgMetaValueExtractor": "Image Load With Meta",
    "SendTelegramChatBot": "Send Image To Telegram Bot",
    "LoraLoaderByPath": "Lora Loader By Path",
    "LoraLoaderOnlyModelByPath": "Lora Loader Only Model By Path",
    "RandomPromptFromBook": "Random Prompt From Book",
}
