import os
import random
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
import copy
import torchvision.transforms.v2 as T

CATEGORY = "SP-Nodes"
MAX_RESOLUTION=16384


class DatasetData:
    def __init__(self, count: int, pct: float, x_minmax: tuple[int, int], y_minmax: tuple[int, int], rotate_degrees_delta: int) -> None:
        self.count: int = count
        self.pct: float = pct
        self.x_minmax: tuple[int, int] = x_minmax
        self.y_minmax: tuple[int, int] = y_minmax
        self.rotate_degrees_delta: int = rotate_degrees_delta

    def __str__(self) -> str:
        return f'count: {self.count}; pct: {self.pct}; x_minmax: {self.x_minmax}; y_minmax: {self.y_minmax}; rotate_degrees_delta: {self.rotate_degrees_delta}'

class ScatterParamsBatchContainer:
    def __init__(self, data) -> None:
        self.data = data

class ScatterParams:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("SCATTER_PARAMS",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "count": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "scale": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "x_min": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "x_max": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "y_min": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "y_max": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "rotate_degrees_delta": ("INT", {"default": 25, "min": 0, "max": 180, "step": 5}),
            },
            "optional": {

            },
        }

        return inputs
    
    def process(self, count, scale, x_min, x_max, y_min, y_max, rotate_degrees_delta):
        return (DatasetData(count, scale, (x_min, x_max), (y_min, y_max), rotate_degrees_delta), )

class ScatterParamsBatch:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("SCATTER_PARAMS",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "scatter_params1": ("SCATTER_PARAMS", ),
                "scatter_params2": ("SCATTER_PARAMS", ),
            },
            "optional": {
                "scatter_params3": ("SCATTER_PARAMS", ),
                "scatter_params4": ("SCATTER_PARAMS", ),
                "scatter_params5": ("SCATTER_PARAMS", ),
                "scatter_params6": ("SCATTER_PARAMS", ),
            },
        }

        return inputs
    
    def process(self, **kwargs):
        out = []
        for v in kwargs.values():
            if isinstance(v, ScatterParamsBatchContainer):
                out.extend(v.data)
            else:
                out.append(v)
        return (ScatterParamsBatchContainer(tuple(out)), )


def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def crop_image_with_transparency_mask(image:Image, mask:Image) -> Image:
    result = Image.new('RGBA', mask.size)
    result.paste(image, mask=mask)

    bbox = mask.getbbox()
    if bbox:
        result = result.crop(bbox)
        
    return result

def get_bbox_with_pixels(image_path):
    img = Image.open(image_path).convert("RGBA")
    return img.crop(img.getbbox())

def calculate_coords(value: tuple[int, int], canvas_size: int, paste_size: int) -> tuple[int, int]:
    min_val = 0                        if value[0] == -1 else max(0, value[0])
    max_val = canvas_size - paste_size if value[1] == -1 else min(canvas_size - paste_size, value[1] - paste_size)

    if min_val > canvas_size - paste_size:
        min_val = canvas_size - paste_size

    if max_val < 0:
        max_val = 0

    return min_val, max_val

def paste_region_random_location(paste_img: Image, canvas_size: tuple[int, int], resize_pct: int, x_minmax: tuple[int, int], y_minmax: tuple[int, int], rotate_degrees_delta: int, allow_flip_images: bool):
    canvas = Image.new('RGBA', canvas_size, color=(0, 0, 0, 0))

    width = int(canvas.width * resize_pct)
    height = int(canvas.height * resize_pct)
    
    if allow_flip_images and random.choice([True, False]):
        paste_img = paste_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    resized_region = paste_img.copy()
    resized_region.thumbnail((width, height))
    
    x_min, x_max = calculate_coords(x_minmax, canvas.width, resized_region.width)
    y_min, y_max = calculate_coords(y_minmax, canvas.height, resized_region.height)

    x = random.randint(min(x_min, x_max), max(x_min, x_max))
    y = random.randint(min(y_min, y_max), max(y_min, y_max))

    rotate_degree = random.randint(-rotate_degrees_delta, rotate_degrees_delta)
    rotated_img = resized_region.rotate(rotate_degree)

    canvas.paste(rotated_img, (x, y))

    return canvas

def pb(image):
    return image.permute([0,2,3,1])

class FaceScatter:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "process"

    def __init__(self):
        pass

    # def __del__(self):
        # if self.observer:
            # self._destroy_observer()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK",),
                "scatter_params": ("SCATTER_PARAMS", ),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {

            },
        }

        return inputs

    

    def process(self, image, mask, scatter_params: DatasetData, width, height):
        image = tensor2pil(torch.unsqueeze(image, 0))
        mask = tensor2pil(torch.unsqueeze(mask, 0)).convert('L')
        face_image = crop_image_with_transparency_mask(image.convert('RGB'), mask)

        scatter_data = [scatter_params] if not isinstance(scatter_params, ScatterParamsBatchContainer) else [param for param in scatter_params.data]
        
        out = []

        for data in scatter_data:
            for i in range(data.count):
                img = paste_region_random_location(face_image, (width, height), data.pct, data.x_minmax, data.y_minmax, data.rotate_degrees_delta)
                out.append(T.ToTensor()(img))
                # out.append(img)

        out = torch.stack(out, dim=0)
        out = pb(out)
        mask = out[:, :, :, 3] if out.shape[3] == 4 else torch.ones_like(out[:, :, :, 0])

        # return (torch.cat(tuple([pil2tensor(i) for i in out]), dim=0), mask, )
        return (out[:, :, :, :3], mask, )
    
class FaceScatter2:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "process"

    def __init__(self):
        pass

    # def __del__(self):
        # if self.observer:
            # self._destroy_observer()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK",),
                "scatter_params": ("SCATTER_PARAMS", ),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "transparency": ("BOOLEAN", {"default": False}),
                "invert_masks": ("BOOLEAN", {"default": True}),
                "allow_flip_images": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {

            },
        }

        return inputs

    

    def process(self, image, mask, scatter_params: DatasetData, width, height, transparency: bool, invert_masks: bool, allow_flip_images: bool, seed):
        random.seed(seed)
        image = tensor2pil(torch.unsqueeze(image, 0))
        mask = tensor2pil(torch.unsqueeze(mask, 0)).convert('L')
        face_image = crop_image_with_transparency_mask(image.convert('RGB'), mask)

        scatter_data = [scatter_params] if not isinstance(scatter_params, ScatterParamsBatchContainer) else [param for param in scatter_params.data]
        
        out = []
        out_masks = []

        for data in scatter_data:
            for i in range(data.count):
                # image
                img = paste_region_random_location(face_image, (width, height), data.pct, data.x_minmax, data.y_minmax, data.rotate_degrees_delta, allow_flip_images)
                # out.append(T.ToTensor()(img))
                out.append(img)

                # masks
                mask_image = img.convert("RGBA")
                r, g, b, a = mask_image.split()

                # if invert_masks:
                #     a = Image.fromarray(255 - np.array(a))
                
                if invert_masks:
                    a = Image.eval(a, lambda x: 255 - x)

                mask_image = pil2tensor(a.convert('L')) # T.ToTensor()(a.convert("L")).permute([0,2,3,1])
                print('shape', mask_image.shape)
                # mask_image = mask_image[:, :, :, 3] if mask_image.shape[3] == 4 else torch.ones_like(mask_image[:, :, :, 0])

                # if invert_masks:
                #     mask_image = 1.0 - mask_image

                out_masks.append(mask_image)

        # out = torch.stack(out, dim=0)
        # out = pb(out)
        # mask = out[:, :, :, 3] if out.shape[3] == 4 else torch.ones_like(out[:, :, :, 0])

        images = torch.cat(tuple([pil2tensor(i if transparency else i.convert("RGB")) for i in out]), dim=0)
        masks = torch.cat(out_masks, dim=0)
        # masks = pb(masks)
        
        # masks = masks[:, :, :, 3] if masks.shape[3] == 4 else torch.ones_like(masks[:, :, :, 0])
        return (images, masks, )
        # return (out[:, :, :, :3], mask, )
        

NODE_CLASS_MAPPINGS = {
    "ScatterParams": ScatterParams,
    "ScatterParamsBatch": ScatterParamsBatch,
    "FaceScatter": FaceScatter,
    "FaceScatter2": FaceScatter2,
}

