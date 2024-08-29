"""
@author: SeniorPioner
@title: SP-Nodes
@nickname: SP-Nodes
@description: Node Pack: PromptChecker for token toggling, KoboldCPP API, ModelMerging, Telegram-Bot-API, and more
"""

import shutil, os, folder_paths

import subprocess
import threading
import locale
import sys
import importlib.util
from .config import write_config

write_config()

WEB_DIRECTORY = "web"
# nodes.EXTENSION_WEB_DIRS["zaio-nodster"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web')
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nodes')

def import_and_merge(file_path):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    module_name = file_path.replace(nodes_directory, '')
    module_name = module_name.replace('\\', '.').replace('/', '.')[:-3]  # Remove .py extension

    try:
        module = importlib.import_module(".nodes{}".format(module_name), __name__)
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"Error importing module {module_name}: {e}")

def find_py_files_and_import(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                import_and_merge(file_path)


find_py_files_and_import(nodes_directory)

cwd_path = os.path.dirname(os.path.realpath(__file__))
comfy_path = folder_paths.base_path

def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)

def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"[ZAIO] EXECUTE: {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()

pip_install = [sys.executable, '-m', 'pip', 'install']

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
except ImportError:
    process_wrap(pip_install + ['nltk'])

    import nltk
    nltk.download('punkt')

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]