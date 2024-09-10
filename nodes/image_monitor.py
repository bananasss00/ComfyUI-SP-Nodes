import os
import numpy as np
from PIL import Image, ImageOps
import torch
from datetime import datetime, timedelta
import sys
import io
import hashlib
import subprocess
import locale
import threading
import zipfile


def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)


def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"[AiO-Node] EXECUTE: {cmd_str} in '{cwd}'")
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


# if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
#     pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
# else:
pip_install = [sys.executable, '-m', 'pip', 'install']

try:
    from psd_tools import PSDImage
except ImportError:
    process_wrap(pip_install + ['--use-pep517', 'psd-tools'])

    from psd_tools import PSDImage

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    process_wrap(pip_install + ['watchdog'])

    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler


CATEGORY = "SP-Nodes"

class ImageMonitor:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    LOADED_IMAGES_HASHES: dict = {}

    def __init__(self):
        self.observer = None
        self.current_image_path = None
        self.image_data = None

    # def __del__(self):
        # if self.observer:
            # self._destroy_observer()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "image_path": ("STRING", {"placeholder": "path_to_image"}),
            },
            "optional": {

            },
        }

        return inputs

    class ImageChangeDetector(FileSystemEventHandler):
        def __init__(self, file_name, callback):
            self.file_name = file_name
            self.callback = callback

        def on_modified(self, event):
            if event.event_type != 'modified':
                return

            if not event.is_directory and event.src_path.lower().endswith(self.file_name.lower()):
                self.callback(event.src_path)

    def process(self, image_path):
        if not self.observer:
            self._load_image_data(image_path)
            self._create_observer(image_path)
            # print('not observer. create')
            
        elif self.current_image_path != image_path:
            self._destroy_observer()
            self._create_observer(image_path)

        image, mask, sha256 = self.image_data
        return (image,)

    def _load_image_data(self, image_path):
        self.image_data = self.load_image(image_path)
        image, mask, sha256 = self.image_data
        # print(f'_load_image_data for {image_path}')
        if not self.observer:
            # bypass dual load image when init workflow
            ImageMonitor.LOADED_IMAGES_HASHES[image_path] = ''
        else:
            ImageMonitor.LOADED_IMAGES_HASHES[image_path] = sha256
            # print(f'new hash {sha256} for {image_path}')

    def _on_file_changed(self, image_path):
        self._print(f'File {image_path} has been modified.')
        self._load_image_data(image_path)

    def _create_observer(self, image_path):
        event_handler = ImageMonitor.ImageChangeDetector(os.path.basename(image_path), self._on_file_changed)
        self.observer = Observer()
        self.observer.schedule(event_handler, path=os.path.dirname(image_path), recursive=False)
        self.observer.start()
        self.current_image_path = image_path
        self._print(' - observer created')

    @classmethod
    def load_image(s, image_path: str):
        with open(image_path, 'rb') as f:
            bytes = f.read()
        sha256 = s.get_sha256(bytes)

        i = None
        if image_path.lower().endswith('.psd'):
            psd = PSDImage.open(io.BytesIO(bytes))
            i = psd.topil()
        elif image_path.lower().endswith('.kra'):
            try:
                with zipfile.ZipFile(image_path, 'r') as zip_file:
                    with zip_file.open('mergedimage.png') as file:
                        bytes = file.read()
                        i = Image.open(io.BytesIO(bytes))
            except Exception as e:
                raise e
        else:
            i = Image.open(io.BytesIO(bytes))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return image, mask, sha256
    
    @classmethod
    def _print(s, text, debug=False):
        if debug:
            with open('a:/image_monitor.log', 'a') as f:
                f.write(f'{text}\n')
        else:
            print(text)

    @classmethod
    def get_sha256(s, bytes):
        m = hashlib.sha256()
        m.update(bytes)
        return m.digest().hex()

    def _destroy_observer(self):
        # TODO: Kill destroyed nodes
        self.observer.stop()
        self.observer.join()
        self.observer = None
        self._print(' - observer destroyed')

    @classmethod
    def IS_CHANGED(s, image_path):
        sha256 = s.LOADED_IMAGES_HASHES.get(image_path, '')
        print(f'{image_path} changed. hash = {sha256}')
        return sha256

    @classmethod
    def VALIDATE_INPUTS(s, image_path):
        if not os.path.isfile(image_path):
            return "Invalid image file: {}".format(image_path)

        return True

NODE_CLASS_MAPPINGS = {
    "ImageMonitor": ImageMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMonitor": "Image Monitor",
}
