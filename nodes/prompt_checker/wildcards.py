# credits: impact-pack nodes
import re
import random
import os
import sys
import yaml
import numpy as np
import threading
# from impact import utils

wildcard_lock = threading.Lock()
wildcard_dict = {}

def get_wildcard_list():
    with wildcard_lock:
        return [f"__{x}__" for x in wildcard_dict.keys()]

def get_wildcard_dict():
    global wildcard_dict
    with wildcard_lock:
        return wildcard_dict
        
def wildcard_normalize(x):
    return x.replace("\\", "/").lower()


def read_wildcard(k, v):
    if isinstance(v, list):
        k = wildcard_normalize(k)
        wildcard_dict[k] = v
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            new_key = f"{k}/{k2}"
            new_key = wildcard_normalize(new_key)
            read_wildcard(new_key, v2)


def read_wildcard_dict(wildcard_path):
    global wildcard_dict
    for root, directories, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, wildcard_path)
                key = os.path.splitext(rel_path)[0].replace('\\', '/').lower()

                try:
                    with open(file_path, 'r', encoding="ISO-8859-1") as f:
                        lines = f.read().splitlines()
                        wildcard_dict[key] = lines
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                        lines = f.read().splitlines()
                        wildcard_dict[key] = lines
            elif file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            
                    for k, v in yaml_data.items():
                        read_wildcard(k, v)

    return wildcard_dict

def process(text, seed=None):
    if seed is not None:
        random.seed(seed)
    random_gen = np.random.default_rng(seed)
    
    def unpack_wildcards(string):
        def unpack_wildcard(match):
            local_wildcard_dict = get_wildcard_dict()
        
            string = match.group(1)
            pattern = r"__([\w.\-+/*\\]+)__"
            matches = re.findall(pattern, string)

            replacements_found = False

            for match in matches:
                keyword = match.lower()
                keyword = wildcard_normalize(keyword)
                if keyword in local_wildcard_dict:
                    replacement = '|'.join(local_wildcard_dict[keyword])
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
                elif '*' in keyword:
                    subpattern = keyword.replace('*', '.*').replace('+','\+')
                    total_patterns = []
                    found = False
                    for k, v in local_wildcard_dict.items():
                        if re.match(subpattern, k) is not None:
                            total_patterns += v
                            found = True

                    if found:
                        replacement = '|'.join(total_patterns)
                        replacements_found = True
                        string = string.replace(f"__{match}__", replacement, 1)
                elif '/' not in keyword:
                    string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                    string = unpack_wildcards(string_fallback)

            return string

        pattern = r'({[^{}]*?})'
        replaced_string = re.sub(pattern, unpack_wildcard, string)

        return replaced_string

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'
            # range_pattern3 = r'^(\d+)'

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])

                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    if r.group(3):
                        b = r.group(3).strip()
                    else:
                        b = a

                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        options[0] = multi_select_pattern[1]
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options[0] = multi_select_pattern[2]

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and is_numeric_string(parts[0].strip()):
                    config_value = float(parts[0].strip())
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                select_count = random_gen.integers(low=select_range[0], high=select_range[1]+1, size=1)

            if select_count > len(options):
                random_gen.shuffle(options)
                selected_items = options
            else:
                selected_items = random_gen.choice(options, p=normalized_probabilities, size=select_count, replace=False)
                selected_items = set(selected_items)  # wildcards here not replaced and removed dublicated

                try_count = 0
                while len(selected_items) < select_count and try_count < 10:
                    remaining_count = select_count - len(selected_items)
                    additional_items = random_gen.choice(options, p=normalized_probabilities, size=select_count, replace=False)
                    selected_items |= set(additional_items)
                    try_count += 1

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', x, 1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        string = unpack_wildcards(string)

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        local_wildcard_dict = get_wildcard_dict()
        pattern = r"__([\w.\-+/*\\]+)__"
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)
            if keyword in local_wildcard_dict:
                replacement = random_gen.choice(local_wildcard_dict[keyword])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+','\+')
                total_patterns = []
                found = False
                for k, v in local_wildcard_dict.items():
                    if re.match(subpattern, k) is not None:
                        total_patterns += v
                        found = True

                if found:
                    replacement = random_gen.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
            elif '/' not in keyword:
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop

        # pass1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # pass2: replace wildcards
        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None

if __name__ == '__main__':
    read_wildcard_dict(r'v:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Impact-Pack\wildcards')

    t1 = '{2$$1|2|3|4|5|6|7|8}'
    t2 = '{2-4$$1|2|3|4|5|6|7|8}'
    t3 = '{-4$$1|2|3|4|5|6|7|8}'

    tt = 'hello world {2-4$$__w2*__} saassss'
    tt2 = '__sk__ {__w1__|__sub/w*__} 1{girl is holding {blue pencil|red {__w*__}|colorful {__w2__}}|boy is riding {2-3$$__q1__}}'
    t4 = 'hello world {2-4$$__cosplay__|__jenres__}'
    t5 = 'hello world {2-4$$__cosplay__|{2-4$$1|2|3|4|5|6}|__jenres__}'
    tfull = '(everything white color:1.2), {0.3::nsfw,|} {__200p/adj/adj-beauty__|__200p/adj/adj-general__|__200p/adj/adj-horror__} {__nation__|__200p/person_tweaks/class/*__|__200p/person_tweaks/occupation__} {woman|girl}, {__200p/person_tweaks/hair/*__}, {1-8$$__cosplay__|__200p/clothes/*__|__clothes*__}, {1-3$$__accessories__}, {__200p/person_tweaks/expression__|__emotions__} emotion, __200p/person_tweaks/eyeliner__, __200p/person_tweaks/lipstick__ __200p/person_tweaks/lipstick-shade__, __200p/person_tweaks/makeup__, __200p/person_tweaks/earrings__, (__200p/person_tweaks/skin-color__ skin:0.1), (__200p/body/*__ body{|0.3::tattoo}:1.15), __200p/person_tweaks/breastsize__, {__200p/scenario/*__|__crazy_actions__|__actions__}, {1-4$$__effects__|__200p/style/*__|__jenres__}, __200p/subject/*__, __200p/time__ , {1-3$$__bg__|__200p/location/*__} background, (by {2-4$$,$$__200p/artist/*__}:1.3)'

    for i in range(10):
        # print(f'=== i = {i} ===')
        # print(process(t1, 1 + i))
        # print(process(t2, 6565 + i))
        # print(process(t3, 432 + i))
        # print(process(t4, 432 + i))
        print(process(tfull, 555 + i))

else:
    # global wildcard_dict
    # sys.path.extend([r'v:\ComfyUI_windows_portable\ComfyUI'])

    # if 'impact.wildcards' in sys.modules:
    #     module = sys.modules['impact.wildcards']
    #     module.process = process
    #     wildcard_dict = module.wildcard_dict
    #     wildcard_lock = module.wildcard_lock

    #     print('== impact.wildcards patched! ==')
    from ...config import read_config
    config = read_config()
    wildcard_dict = read_wildcard_dict(config['wildcards_path'])
    
def test():
    pass
    
