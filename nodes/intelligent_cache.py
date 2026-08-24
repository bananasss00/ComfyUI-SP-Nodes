import hashlib
import logging
import os
import re
import sys
import joblib
from comfy.comfy_types import IO
import numpy as np
import torch
import server
from aiohttp import web
import threading

# Set up a logger for the nodes
logger = logging.getLogger(__name__)

# A global dictionary to serve as an in-memory cache for all nodes in this file.
CACHE = {}

def _background_save(data, filepath, compression_level):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # joblib.dump теперь использует параметр compress
        joblib.dump(data, filepath, compress=compression_level)
        logger.info(f"Background save completed: {filepath} (Compression: {compression_level})")
    except Exception as e:
        logger.error(f"Background save failed for {filepath}: {e}")
        
class Color:
    """A simple class for adding color to console output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    @staticmethod
    def green(s):
        return f"{Color.GREEN}{s}{Color.RESET}"

    @staticmethod
    def yellow(s):
        return f"{Color.YELLOW}{s}{Color.RESET}"

    @staticmethod
    def blue(s):
        return f"{Color.BLUE}{s}{Color.RESET}"

# --- Helper Functions for Memory Calculation ---

def format_size(size_bytes):
    """Converts bytes to a human-readable string (B, KB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    if i >= len(size_name):
        i = len(size_name) - 1
    return f"{s} {size_name[i]}"

def get_deep_size(obj, seen=None):
    """Recursively finds the size of objects including Tensors and Arrays."""
    size = 0
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    try:
        if isinstance(obj, torch.Tensor):
            size += obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            size += obj.nbytes
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sys.getsizeof(obj)
            for item in obj:
                size += get_deep_size(item, seen)
        elif isinstance(obj, dict):
            size += sys.getsizeof(obj)
            for k, v in obj.items():
                size += get_deep_size(k, seen)
                size += get_deep_size(v, seen)
        elif hasattr(obj, '__dict__'):
             size += sys.getsizeof(obj)
             size += get_deep_size(obj.__dict__, seen)
        else:
            size += sys.getsizeof(obj)
    except Exception:
        size += sys.getsizeof(obj)
        
    return size

class SP_CacheValue:
    """
    Caches a value based on a key.
    If the key is found and 'overwrite' is False, it returns the cached value.
    Otherwise, it evaluates the 'value' input, caches it, and returns it.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_key", "tooltip": "The unique key for this cache entry."}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "If True, always re-evaluate the input and overwrite the existing cache entry."}),
            },
            "optional": {
                "value": (IO.ANY, {"lazy": True, "tooltip": "The value to cache. This is only evaluated on a cache miss or when 'overwrite' is True."}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "get_or_cache_value"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Caches a value. If 'overwrite' is True, it acts like CacheStore."

    def check_lazy_status(self, key, overwrite, value=None):
        if value is not None:
            return []
        # If overwrite is enabled, we always need the value to be computed.
        if overwrite:
            return ["value"]
        # Otherwise, we only need it if it's a cache miss.
        if key not in CACHE:
            return ["value"]
        return []

    def get_or_cache_value(self, key, overwrite, value=None):
        # If not overwriting and the key is in the cache, return the cached value.
        if not overwrite and key in CACHE:
            return ([CACHE[key]],)
        else:
            # This block is reached if overwrite is True or if it's a cache miss.
            if value is None:
                error_message = f"Cache MISS or OVERWRITE for key '{key}', but no input was provided to 'value' to create the cache entry."
                logger.error(error_message)
                raise ValueError(error_message)
            CACHE[key] = value
            return ([value],)

class SP_CacheManager:
    """
    A utility node to manage the global cache. It can view the contents,
    clear a specific key, or clear the entire cache with memory stats.
    """
    ACTION_LIST = ["view", "clear_key", "clear_prefix", "clear_all"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (cls.ACTION_LIST, {"tooltip": "The management action to perform on the cache."}),
            },
            "optional": {
                "key": ("STRING", {"default": "default_key", "tooltip": "The specific key to clear. Only used with the 'clear_key' action."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_info",)
    FUNCTION = "manage_cache"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "View, clear a key, or clear the entire cache."

    def manage_cache(self, action, key=None):
        output_message = ""
        
        if action == "view":
            if not CACHE: 
                output_message = "Cache is currently empty."
            else:
                total_mem = 0
                output_message = f"Cache Contents ({len(CACHE)} items):\n"
                output_message += "=" * 40 + "\n\n"
                
                for k, v in CACHE.items():
                    item_size = get_deep_size(v)
                    total_mem += item_size
                    readable_size = format_size(item_size)
                    
                    val_str = str(v).replace('\n', ' ')
                    if len(val_str) > 100: val_str = val_str[:100] + "..."
                    
                    output_message += f"🔑 Key: '{k}'\n"
                    output_message += f"📦 Size: {readable_size}\n"
                    output_message += f"📄 Type: {type(v).__name__}\n"
                    output_message += f"📝 Value: {val_str}\n"
                    output_message += "-" * 20 + "\n\n"
                
                output_message += "=" * 40 + "\n"
                output_message += f"Total Cache Memory: {format_size(total_mem)}"

        elif action == "clear_key":
            if key is None or key.strip() == "": 
                output_message = "Error: A valid key must be provided to clear a single entry."
            elif key in CACHE:
                freed_size = format_size(get_deep_size(CACHE[key]))
                del CACHE[key]
                output_message = f"Successfully cleared key: '{key}'. Freed: {freed_size}"
            else: 
                output_message = f"Key '{key}' not found in cache."

        elif action == "clear_prefix":
            if not key:
                return ("Error: Provide a prefix in the 'key' field.",)
            keys_to_delete = [k for k in CACHE.keys() if k.startswith(key)]
            freed_size = sum(get_deep_size(CACHE[k]) for k in keys_to_delete)
            for k in keys_to_delete:
                del CACHE[k]
            output_message = f"Cleared {len(keys_to_delete)} items with prefix '{key}'. Freed: {format_size(freed_size)}"

        elif action == "clear_all":
            num_items = len(CACHE)
            if num_items > 0:
                # Считаем общий размер перед удалением
                total_bytes = sum(get_deep_size(v) for v in CACHE.values())
                freed_str = format_size(total_bytes)
                CACHE.clear()
                output_message = f"Successfully cleared entire cache ({num_items} items).\nFreed Memory: {freed_str}"
            else:
                output_message = "Cache is already empty."
            
        return (output_message,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class SP_CacheCheck:
    """
    Checks if a specific key exists in the global cache and returns a boolean.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_key", "tooltip": "The key to check for existence in the cache."}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_cached",)
    FUNCTION = "check_cache"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Returns True if the key is cached, otherwise False."

    def check_cache(self, key):
        return (key in CACHE,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class SP_CacheStore:
    """
    Unconditionally stores or overwrites a value in the cache with a given key.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_key", "tooltip": "The unique key to store the value under. Will overwrite if it already exists."}),
                "value": (IO.ANY, {"tooltip": "The value to be stored in the cache."}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("value",)
    OUTPUT_NODE = True
    FUNCTION = "store_value"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Forcibly stores or overwrites a value in the cache."

    def store_value(self, key, value):
        CACHE[key] = value
        logger.info(f"Stored/updated value for key '{key}' in cache.")
        return (value,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class SP_CacheGet:
    """
    Retrieves a value from the cache. If the key is not found, it returns
    a provided default value instead of failing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_key", "tooltip": "The key of the value to retrieve from the cache."}),
            },
            "optional": {
                "default_value": (IO.ANY, {"tooltip": "The value to return if the key is not found in the cache."}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "get_value"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Gets a value from cache; returns a default if not found."

    def get_value(self, key, default_value=None):
        value = CACHE.get(key, default_value)
        return ([value],)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class SP_CompositeCacheKey:
    """
    Constructs a single key string from multiple parts, joined by a separator.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "", "tooltip": "The initial part of the key."}),
                "separator": ("STRING", {"default": "_", "tooltip": "The character or string used to join the key parts."}),
            },
            "optional": {
                "part_A": ("STRING", {"default": "", "tooltip": "An optional middle part of the key."}),
                "part_B": ("STRING", {"default": "", "tooltip": "Another optional middle part of the key."}),
                "part_C": ("STRING", {"default": "", "tooltip": "An optional final part of the key."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("key",)
    FUNCTION = "create_key"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Creates a composite key from multiple string parts."

    def create_key(self, prefix, separator, part_A=None, part_B=None, part_C=None):
        parts = [str(p) for p in [prefix, part_A, part_B, part_C] if p is not None and str(p).strip() != ""]
        composite_key = separator.join(parts)
        return (composite_key,)
    
class SP_SmartHashKey:
    """
    Takes up to 8 inputs, automatically combines them to generate a short MD5 hash key.
    Passes all inputs through to their respective outputs to keep the graph linear and clean!
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "job", "tooltip": "Prefix for the generated hash string."}),
            },
            "optional": {
                "arg_1": (IO.ANY,),
                "arg_2": (IO.ANY,),
                "arg_3": (IO.ANY,),
                "arg_4": (IO.ANY,),
                "arg_5": (IO.ANY,),
                "arg_6": (IO.ANY,),
                "arg_7": (IO.ANY,),
                "arg_8": (IO.ANY,),
            }
        }

    # Возвращаем строку (хеш) + 8 проброшенных значений
    RETURN_TYPES = ("STRING", IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY,)
    RETURN_NAMES = ("hash_key", "arg_1", "arg_2", "arg_3", "arg_4", "arg_5", "arg_6", "arg_7", "arg_8",)
    FUNCTION = "generate_hash"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Generates a hash key and passes arguments through."

    def generate_hash(self, prefix, arg_1=None, arg_2=None, arg_3=None, arg_4=None, arg_5=None, arg_6=None, arg_7=None, arg_8=None):
        kwargs = {
            "arg_1": arg_1, "arg_2": arg_2, "arg_3": arg_3, "arg_4": arg_4,
            "arg_5": arg_5, "arg_6": arg_6, "arg_7": arg_7, "arg_8": arg_8,
        }
        
        combined_string = ""
        for i in range(1, 9):
            val = kwargs[f"arg_{i}"]
            if val is not None:
                combined_string += str(val) + "|"

        md5_hash = hashlib.md5(combined_string.encode('utf-8')).hexdigest()
        short_hash = f"{prefix}_{md5_hash[:10]}"
        
        # Возвращаем кортеж: сначала хеш, потом все проброшенные аргументы без изменений
        return (short_hash, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8,)

    @classmethod
    def IS_CHANGED(cls, prefix, arg_1=None, arg_2=None, arg_3=None, arg_4=None, arg_5=None, arg_6=None, arg_7=None, arg_8=None):
        kwargs = {
            "arg_1": arg_1, "arg_2": arg_2, "arg_3": arg_3, "arg_4": arg_4,
            "arg_5": arg_5, "arg_6": arg_6, "arg_7": arg_7, "arg_8": arg_8,
        }
        combined_string = ""
        for i in range(1, 9):
            val = kwargs[f"arg_{i}"]
            if val is not None:
                combined_string += str(val) + "|"
        return hashlib.md5(combined_string.encode('utf-8')).hexdigest()

class SP_CachePersistence:
    """
    Saves the in-memory cache to a file on disk or loads it back.
    Uses joblib for efficient object serialization.
    """
    ACTION_LIST = ["save_to_disk", "load_from_disk", "merge_from_disk"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (cls.ACTION_LIST, {"tooltip": "The persistence action to perform."}),
                "filepath": ("STRING", {"default": "comfyui_cache.joblib", "tooltip": "The full path to the file for saving or loading the cache."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "persist_cache"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Save cache to disk or load it back using joblib."

    def persist_cache(self, action, filepath):
        try:
            if action == "save_to_disk":
                dir_name = os.path.dirname(filepath)
                if dir_name: os.makedirs(dir_name, exist_ok=True)
                joblib.dump(CACHE, filepath)
                return (f"Successfully saved {len(CACHE)} items to {filepath}",)

            if not os.path.exists(filepath): return (f"Error: File not found at {filepath}",)

            if action == "load_from_disk":
                loaded_cache = joblib.load(filepath)
                CACHE.clear()
                CACHE.update(loaded_cache)
                return (f"Successfully cleared and loaded {len(CACHE)} items from {filepath}",)

            if action == "merge_from_disk":
                loaded_cache = joblib.load(filepath)
                CACHE.update(loaded_cache)
                return (f"Successfully merged data. Cache now has {len(CACHE)} items.",)
        except Exception as e:
            logger.error(f"Cache persistence error: {e}")
            return (f"An error occurred: {e}",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class SP_CacheAutoLoader:
    """
    A smart cache node that uses a three-tier system: memory -> disk -> compute.
    It automatically handles loading from disk if not in memory, and computes/saves
    the value only if it's not found in either cache.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_key", "tooltip": "The unique key for the cache entry."}),
                "cache_directory": ("STRING", {"default": "sp_node_cache", "tooltip": "The directory on disk where the cache file will be stored."}),
                "compression_level": ("INT", {"default": 0, "min": 0, "max": 9, "tooltip": "0 = Fastest but large files. 3 = Balanced. 9 = Smallest files but slow."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "If False, the node will simply pass the value through without any caching."}),
                "force_recompute": ("BOOLEAN", {"default": False, "tooltip": "If True, ignores cache completely, re-evaluates input, and overwrites the disk file."}),
                "unload_from_ram": ("BOOLEAN", {"default": False, "tooltip": "If True, removes the value from RAM after loading/caching, keeping it ONLY on disk. Great for cross-workflow caching."}),
            },
            "optional": {
                "value": (IO.ANY, {"lazy": True, "tooltip": "The value to cache. This is only evaluated if the key is not found in memory or on disk."}),
            }
        }

    RETURN_TYPES = (IO.ANY, "STRING",)
    RETURN_NAMES = ("value", "status",)
    OUTPUT_IS_LIST = (True, False,)
    FUNCTION = "load_or_compute"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Auto-caches to memory and disk; computes only if necessary."

    def _get_filepath(self, cache_directory, key):
        """Helper method to construct the full file path for a given key."""
        safe_filename = "".join(c for c in key if c.isalnum() or c in ('_', '-', '.'))
        filename = f"{safe_filename}.joblib"
        return os.path.join(cache_directory, filename)

    def check_lazy_status(self, key, cache_directory, compression_level, enabled, force_recompute, unload_from_ram, **kwargs):
        connected_values = [k for k in kwargs.keys() if k == "value"]
        
        # Если кэш выключен, нам всегда нужно значение (если оно подключено)
        if not enabled: 
            return connected_values

        # Если принудительный пересчет - всегда требуем вычислить значение
        if force_recompute:
            return connected_values

        # Иначе вычисляем только если нет ни в ОЗУ, ни на диске
        filepath = self._get_filepath(cache_directory, key)
        if key not in CACHE and not os.path.exists(filepath):
            return connected_values

        return []

    def load_or_compute(self, key, cache_directory, compression_level, enabled, force_recompute, unload_from_ram, value=None):
        """
        Loads data from cache (memory/disk) or computes it if not found.
        """
        # 1. Режим выключенного кэша
        if not enabled:
            if value is None:
                raise ValueError(f"Caching is disabled for key '{key}', but no input was provided to 'value'.")
            return ([value], "Cache disabled; value passed through",)

        # 2. Принудительный пересчет (Force Recompute)
        if force_recompute:
            if value is None:
                raise ValueError(f"Force recompute is ON for key '{key}', but no input was provided to 'value'.")

            # Сохраняем на диск в фоновом потоке
            filepath = self._get_filepath(cache_directory, key)
            try:
                os.makedirs(cache_directory, exist_ok=True)
                threading.Thread(target=_background_save, args=(value, filepath, compression_level), daemon=True).start()
            except Exception as e:
                logger.error(f"Failed to save cache file to {filepath}: {e}")

            # Если стоит галочка не держать в ОЗУ - не записываем в CACHE
            if not unload_from_ram:
                CACHE[key] = value
            elif key in CACHE:
                # На всякий случай очищаем старое значение из ОЗУ, если оно там было
                del CACHE[key]

            return ([value], "Forced recompute; saved to disk",)

        # 3. Обычный режим: проверяем ОЗУ
        if key in CACHE:
            val = CACHE[key]
            # Если стоит галочка выгрузить из ОЗУ - удаляем из словаря
            if unload_from_ram:
                del CACHE[key]
                return ([val], "Loaded from memory; unloaded from RAM",)
            return ([val], "Loaded from memory",)

        # 4. Обычный режим: проверяем Диск
        filepath = self._get_filepath(cache_directory, key)
        if os.path.exists(filepath):
            try:
                loaded_value = joblib.load(filepath)
                # Записываем в ОЗУ только если нет галочки unload_from_ram
                if not unload_from_ram:
                    CACHE[key] = loaded_value
                return ([loaded_value], "Loaded from disk" + (" (RAM bypass)" if unload_from_ram else ""),)
            except Exception as e:
                logger.error(f"Failed to load cache file {filepath}, will re-compute. Error: {e}")

        # 5. Обычный режим: Промах кэша (Cache MISS)
        if value is None:
            raise ValueError(f"Cache MISS for key '{key}' in memory and on disk, but no input was provided to 'value'.")

        # Сохраняем на диск
        try:
            os.makedirs(cache_directory, exist_ok=True)
            threading.Thread(target=_background_save, args=(value, filepath, compression_level), daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to save cache file to {filepath}: {e}")

        # Сохраняем в ОЗУ только если нет галочки unload_from_ram
        if not unload_from_ram:
            CACHE[key] = value

        return ([value], "Newly computed and cached" + (" (RAM bypass)" if unload_from_ram else ""),)

class SP_CacheAutoLoaderMulti:
    """
    A smart cache node for MULTIPLE values. Uses a three-tier system: memory -> disk -> compute.
    Stores and retrieves up to 5 values under a single dictionary key.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_multi_key", "tooltip": "The unique key for the cache entry."}),
                "cache_directory": ("STRING", {"default": "sp_node_cache", "tooltip": "The directory on disk where the cache file will be stored."}),
                "compression_level": ("INT", {"default": 0, "min": 0, "max": 9, "tooltip": "0 = Fastest but large files. 3 = Balanced. 9 = Smallest files but slow."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "If False, the node will simply pass the values through without any caching."}),
            },
            "optional": {
                "value_1": (IO.ANY, {"lazy": True, "tooltip": "Value 1 to cache."}),
                "value_2": (IO.ANY, {"lazy": True, "tooltip": "Value 2 to cache."}),
                "value_3": (IO.ANY, {"lazy": True, "tooltip": "Value 3 to cache."}),
                "value_4": (IO.ANY, {"lazy": True, "tooltip": "Value 4 to cache."}),
                "value_5": (IO.ANY, {"lazy": True, "tooltip": "Value 5 to cache."}),
            }
        }

    RETURN_TYPES = (IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, "STRING",)
    RETURN_NAMES = ("value_1", "value_2", "value_3", "value_4", "value_5", "status",)
    # Оставляем списки (True) для значений, как в оригинале, и False для статуса
    OUTPUT_IS_LIST = (True, True, True, True, True, False,) 
    FUNCTION = "load_or_compute"
    CATEGORY = 'SP-Nodes/cache'
    DESCRIPTION = "Auto-caches MULTIPLE values to memory and disk; computes only if necessary."

    def _get_filepath(self, cache_directory, key):
        """Helper method to construct the full file path. Added '_multi' to prevent collisions."""
        safe_filename = "".join(c for c in key if c.isalnum() or c in ('_', '-', '.'))
        filename = f"{safe_filename}_multi.joblib"
        return os.path.join(cache_directory, filename)

    def _get_memory_key(self, key):
        """Prefix the key for memory cache to avoid collision with single-value cache node."""
        return f"multi_cache_{key}"

    def check_lazy_status(self, key, cache_directory, compression_level, enabled, **kwargs):
        """
        Dynamically checks which inputs are actually connected and only requests those.
        Fixes the 'Node says it needs input, but there is no input' ComfyUI crash.
        """
        # kwargs содержит только те пины, к которым реально подключен провод
        connected_values = [k for k in kwargs.keys() if k.startswith("value_")]

        if not enabled:
            return connected_values

        mem_key = self._get_memory_key(key)
        filepath = self._get_filepath(cache_directory, key)
        
        if mem_key not in CACHE and not os.path.exists(filepath):
            return connected_values

        return[]

    def load_or_compute(self, key, cache_directory, compression_level, enabled, value_1=None, value_2=None, value_3=None, value_4=None, value_5=None):
        """
        Loads multiple data from cache (memory/disk) or computes it if not found.
        """
        mem_key = self._get_memory_key(key)
        
        # Упаковываем текущие значения в словарь
        current_values = {
            "value_1": value_1,
            "value_2": value_2,
            "value_3": value_3,
            "value_4": value_4,
            "value_5": value_5,
        }

        # 1. Если отключено - просто пробрасываем
        if not enabled:
            return ([value_1], [value_2], [value_3], [value_4],[value_5], "Cache disabled; values passed through",)

        # 2. Проверяем память
        if mem_key in CACHE:
            cached = CACHE[mem_key]
            return ([cached.get("value_1")], [cached.get("value_2")], [cached.get("value_3")], 
                [cached.get("value_4")], [cached.get("value_5")], "Loaded from memory",
            )

        # 3. Проверяем диск
        filepath = self._get_filepath(cache_directory, key)
        if os.path.exists(filepath):
            try:
                cached = joblib.load(filepath)
                CACHE[mem_key] = cached
                return (
                    [cached.get("value_1")], [cached.get("value_2")],[cached.get("value_3")], 
                    [cached.get("value_4")], [cached.get("value_5")], "Loaded from disk",
                )
            except Exception as e:
                logger.error(f"Failed to load multi-cache file {filepath}, will re-compute. Error: {e}")

        # 4. Промах кеша (MISS) - сохраняем новые значения
        CACHE[mem_key] = current_values

        try:
            os.makedirs(cache_directory, exist_ok=True)
            threading.Thread(target=_background_save, args=(current_values, filepath, compression_level), daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to save multi-cache file to {filepath}: {e}")

        return ([value_1], [value_2],[value_3], [value_4], [value_5], "Newly computed and cached",)

class SP_CacheAutoLoaderMulti10:
    """
    A massive smart cache node for up to 10 values. 
    Uses a three-tier system: memory -> disk -> compute with background saving.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_multi10_key"}),
                "cache_directory": ("STRING", {"default": "sp_node_cache"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "compression_level": ("INT", {"default": 0, "min": 0, "max": 9, "tooltip": "0: Fast save/load. 3-9: Saves disk space."}),
            },
            "optional": {
                "value_1": (IO.ANY, {"lazy": True}),
                "value_2": (IO.ANY, {"lazy": True}),
                "value_3": (IO.ANY, {"lazy": True}),
                "value_4": (IO.ANY, {"lazy": True}),
                "value_5": (IO.ANY, {"lazy": True}),
                "value_6": (IO.ANY, {"lazy": True}),
                "value_7": (IO.ANY, {"lazy": True}),
                "value_8": (IO.ANY, {"lazy": True}),
                "value_9": (IO.ANY, {"lazy": True}),
                "value_10": (IO.ANY, {"lazy": True}),
            }
        }

    RETURN_TYPES = (IO.ANY, ) * 10 + ("STRING",)
    RETURN_NAMES = ("value_1", "value_2", "value_3", "value_4", "value_5", 
                    "value_6", "value_7", "value_8", "value_9", "value_10", "status",)
    OUTPUT_IS_LIST = (True,) * 10 + (False,) 
    FUNCTION = "load_or_compute"
    CATEGORY = 'SP-Nodes/cache'

    def _get_filepath(self, cache_directory, key):
        safe_filename = "".join(c for c in key if c.isalnum() or c in ('_', '-', '.'))
        return os.path.join(cache_directory, f"{safe_filename}_multi10.joblib")

    def _get_memory_key(self, key):
        return f"multi10_cache_{key}"

    def check_lazy_status(self, key, cache_directory, enabled, compression_level, **kwargs):
        # Только подключенные входы
        connected_values =[k for k in kwargs.keys() if k.startswith("value_")]

        if not enabled:
            return connected_values

        mem_key = self._get_memory_key(key)
        filepath = self._get_filepath(cache_directory, key)
        
        if mem_key not in CACHE and not os.path.exists(filepath):
            return connected_values

        return[]

    def load_or_compute(self, key, cache_directory, enabled, compression_level, 
                        value_1=None, value_2=None, value_3=None, value_4=None, value_5=None,
                        value_6=None, value_7=None, value_8=None, value_9=None, value_10=None):
        mem_key = self._get_memory_key(key)
        
        # Собираем все 10 входов
        current_values = {
            "value_1": value_1, "value_2": value_2, "value_3": value_3, "value_4": value_4, "value_5": value_5,
            "value_6": value_6, "value_7": value_7, "value_8": value_8, "value_9": value_9, "value_10": value_10,
        }

        if not enabled:
            return ([value_1],[value_2], [value_3], [value_4], [value_5], 
                    [value_6],[value_7], [value_8], [value_9], [value_10], "Cache disabled")

        if mem_key in CACHE:
            c = CACHE[mem_key]
            return ([c.get("value_1")], [c.get("value_2")],[c.get("value_3")], [c.get("value_4")], [c.get("value_5")],[c.get("value_6")], [c.get("value_7")],[c.get("value_8")], [c.get("value_9")], [c.get("value_10")], "Loaded from memory")

        filepath = self._get_filepath(cache_directory, key)
        if os.path.exists(filepath):
            try:
                c = joblib.load(filepath)
                CACHE[mem_key] = c
                return ([c.get("value_1")],[c.get("value_2")], [c.get("value_3")], [c.get("value_4")],[c.get("value_5")],
                        [c.get("value_6")],[c.get("value_7")], [c.get("value_8")], [c.get("value_9")],[c.get("value_10")], "Loaded from disk")
            except Exception as e:
                logger.error(f"Multi10-cache load failed {filepath}: {e}")

        CACHE[mem_key] = current_values

        # Фоновое сохранение (не вешает UI)
        threading.Thread(target=_background_save, args=(current_values, filepath, compression_level), daemon=True).start()

        return ([value_1], [value_2], [value_3], [value_4], [value_5],
                [value_6], [value_7], [value_8], [value_9],[value_10], "Computed and background-saving")
    
# --- Web API Endpoints for Cache Management ---

@server.PromptServer.instance.routes.get("/sp_nodes/cache/clear_all")
async def clear_all_cache(request):
    num_items = len(CACHE)
    if num_items > 0:
        total_bytes = sum(get_deep_size(v) for v in CACHE.values())
        readable_size = format_size(total_bytes)
        
        CACHE.clear()
        
        message = f"Successfully cleared the entire cache ({num_items} items). Freed: {readable_size}"
    else:
        message = "Cache is already empty."

    print(Color.green(f"[SP_Nodes] {message}"))
    return web.json_response({"status": "success", "message": message})

@server.PromptServer.instance.routes.get("/sp_nodes/cache/clear/{key}")
async def clear_cache_by_key(request):
    key = request.match_info.get("key", None)
    if key and key in CACHE:
        freed_size = format_size(get_deep_size(CACHE[key]))
        del CACHE[key]
        
        message = f"Successfully cleared cache for key: '{key}'. Freed: {freed_size}"
        print(Color.green(f"[SP_Nodes] {message}"))
        return web.json_response({"status": "success", "message": message})
    else:
        message = f"Key '{key}' not found in cache. Nothing to clear."
        print(Color.yellow(f"[SP_Nodes] {message}"))
        return web.json_response({"status": "not_found", "message": message}, status=404)

@server.PromptServer.instance.routes.get("/sp_nodes/cache/view")
async def view_cache(request):
    if not CACHE:
        message = "Cache is currently empty."
        print(Color.yellow(f"[SP_Nodes] {message}"))
        return web.json_response({"status": "empty", "cache": {}})

    response_data = {}
    total_mem = 0
    
    print(Color.blue(f"\n--- [SP_Nodes] Cache Contents ({len(CACHE)} items) ---"))
    for k, v in CACHE.items():
        size_bytes = get_deep_size(v)
        total_mem += size_bytes
        readable_size = format_size(size_bytes)
        
        val_str = str(v).replace('\n', ' ')
        if len(val_str) > 150:
            val_str = val_str[:150] + "..."
        
        response_data[k] = {
            "type": type(v).__name__,
            "value": val_str,
            "size": readable_size
        }
        
        print(f"- {Color.green('Key')}: '{k}'")
        print(f"  {Color.yellow('Size')}: {readable_size}")
        print(f"  {Color.yellow('Type')}: {type(v).__name__}\n")
        
    total_str = format_size(total_mem)
    print(Color.blue(f"--- Total Memory: {total_str} ---"))
    print(Color.blue("--- End of Cache Contents ---\n"))
    
    # Добавляем поле total_memory в ответ JSON, чтобы фронтенд мог его показать
    return web.json_response({
        "status": "success",
        "total_memory": total_str,
        "items": response_data
    })


class SP_CacheRouter:
    """
    Logical IF/ELSE gate. Checks if a cache exists for the given key.
    Uses Lazy Evaluation so it ONLY computes the branch that is actually needed.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "default_key"}),
                "cache_directory": ("STRING", {"default": "sp_node_cache"}),
            },
            "optional": {
                "if_cached_flow": (IO.ANY, {"lazy": True, "tooltip": "Evaluated and passed ONLY if the cache EXISTS."}),
                "if_not_cached_flow": (IO.ANY, {"lazy": True, "tooltip": "Evaluated and passed ONLY if the cache is MISSING."}),
            }
        }

    RETURN_TYPES = (IO.ANY, "STRING",)
    RETURN_NAMES = ("routed_output", "status",)
    OUTPUT_IS_LIST = (True, False,)
    FUNCTION = "route_flow"
    CATEGORY = 'SP-Nodes/cache/logic'
    DESCRIPTION = "Routes execution based on whether cache exists."

    def _check_exists(self, key, cache_directory):
        safe_filename = "".join(c for c in key if c.isalnum() or c in ('_', '-', '.'))
        
        # 1. Проверка одиночного кеша
        file_single = os.path.join(cache_directory, f"{safe_filename}.joblib")
        if key in CACHE or os.path.exists(file_single):
            return True
            
        # 2. Проверка мульти-кеша
        mem_key_multi = f"multi_cache_{key}"
        file_multi = os.path.join(cache_directory, f"{safe_filename}_multi.joblib")
        if mem_key_multi in CACHE or os.path.exists(file_multi):
            return True
            
        return False

    def check_lazy_status(self, key, cache_directory, if_cached_flow=None, if_not_cached_flow=None):
        if if_cached_flow is not None or if_not_cached_flow is not None:
            return[]
            
        if self._check_exists(key, cache_directory):
            return ["if_cached_flow"]
        else:
            return["if_not_cached_flow"]

    def route_flow(self, key, cache_directory, if_cached_flow=None, if_not_cached_flow=None):
        if if_cached_flow is not None:
            return ([if_cached_flow], "Routed -> CACHED FLOW")
        elif if_not_cached_flow is not None:
            return ([if_not_cached_flow], "Routed -> NOT CACHED FLOW")
        else:
            # Предохранитель, если к ноде ничего не подключили
            return ([None], "No flows connected")
        
class SP_DirToKeys:
    """
    Scans a directory for files (e.g., your selected .mp4 videos),
    cleans up their names (removes extensions and ComfyUI '_00001' counters),
    and outputs a LIST of clean cache keys.
    ComfyUI will automatically batch-execute the rest of the workflow for each key!
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "output/selected", "tooltip": "Folder with your sorted/selected videos."}),
                "file_extension": ("STRING", {"default": ".mp4", "tooltip": "Filter by extension. Use '*' for all files."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("keys_list",)
    # OUTPUT_IS_LIST = True - это магия! Она превращает ноду в генератор батчей для всего графа.
    OUTPUT_IS_LIST = (True,) 
    FUNCTION = "get_keys"
    CATEGORY = 'SP-Nodes/cache/utils'
    DESCRIPTION = "Batch loads cache keys from selected video files."

    def get_keys(self, directory_path, file_extension):
        # Resolve relative paths (e.g., relative to ComfyUI base dir)
        if not os.path.isabs(directory_path):
            directory_path = os.path.join(os.getcwd(), directory_path)

        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return ([],)
            
        keys =[]
        for f in os.listdir(directory_path):
            if file_extension != "*" and not f.endswith(file_extension):
                continue
                
            # 1. Удаляем расширение (.mp4, .png и т.д.)
            name = os.path.splitext(f)[0]
            
            # 2. Удаляем счетчик ComfyUI (например, _00001, _00002). 
            # Ищет нижнее подчеркивание и от 3 до 6 цифр в самом конце строки.
            name = re.sub(r'_\d{3,6}_?$', '', name)
            
            # 3. На случай, если вы натравили ноду прямо на папку с кешем, отрезаем и _multi
            if name.endswith('_multi'):
                name = name[:-6]
                
            if name not in keys:
                keys.append(name)
                
        if not keys:
            logger.warning(f"No matching files found to extract keys in: {directory_path}")
        else:
            logger.info(f"Extracted {len(keys)} keys from {directory_path}: {keys}")
            
        return (keys,)

class SP_CleanKeyString:
    """
    A utility node that takes a single filename/string from ANY other node,
    removes the extension and ComfyUI counters, and outputs a clean cache key.
    Useful if you use external list nodes or VHS Video Loader.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_string": ("STRING", {"forceInput": True, "tooltip": "Connect a string or filename here."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("clean_key",)
    FUNCTION = "clean"
    CATEGORY = 'SP-Nodes/cache/utils'
    DESCRIPTION = "Cleans a filename string to match original cache key."

    def clean(self, filename_string):
        name = os.path.splitext(filename_string)[0]
        name = re.sub(r'_\d{3,6}_?$', '', name)
        if name.endswith('_multi'):
            name = name[:-6]
        return (name,)
    

class SP_DataPacker:
    """
    Packs up to 8 any type of inputs into a single data object.
    Perfect for bypassing slot limits: pack 8 items, send them through ONE cache slot, 
    and unpack on the other side.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "item_1": (IO.ANY, {"tooltip": "Any data to pack."}),
                "item_2": (IO.ANY,),
                "item_3": (IO.ANY,),
                "item_4": (IO.ANY,),
                "item_5": (IO.ANY,),
                "item_6": (IO.ANY,),
                "item_7": (IO.ANY,),
                "item_8": (IO.ANY,),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("packed_data",)
    FUNCTION = "pack"
    CATEGORY = 'SP-Nodes/cache/utils'
    DESCRIPTION = "Packs multiple variables into a single wire."

    def pack(self, **kwargs):
        # kwargs будет содержать только те item_X, которые реально подключены
        return (kwargs,)

class SP_DataUnpacker:
    """
    Unpacks a data object created by SP_DataPacker back into 8 separate wires.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "packed_data": (IO.ANY, {"tooltip": "Connect the packed data wire here."}),
            }
        }

    RETURN_TYPES = (IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY, IO.ANY,)
    RETURN_NAMES = ("item_1", "item_2", "item_3", "item_4", "item_5", "item_6", "item_7", "item_8",)
    FUNCTION = "unpack"
    CATEGORY = 'SP-Nodes/cache/utils'
    DESCRIPTION = "Unpacks packed variables back into separate wires."

    def unpack(self, packed_data):
        if not isinstance(packed_data, dict):
            logger.error("SP_DataUnpacker received invalid data. Expected a packed dictionary from SP_DataPacker.")
            return tuple(None for _ in range(8))
            
        # Аккуратно достаем каждое значение, если его нет - вернется None (что абсолютно нормально)
        return tuple(packed_data.get(f"item_{i}") for i in range(1, 9))
    
NODE_CLASS_MAPPINGS = {
   "SP_CacheValue": SP_CacheValue,
   "SP_CacheManager": SP_CacheManager,
   "SP_CacheCheck": SP_CacheCheck,
   "SP_CacheStore": SP_CacheStore,
   "SP_CacheGet": SP_CacheGet,
   "SP_CacheCompositeKey": SP_CompositeCacheKey,
   "SP_CacheSmartHashKey": SP_SmartHashKey,
   "SP_CachePersistence": SP_CachePersistence,
   "SP_CacheAutoLoader": SP_CacheAutoLoader,
   "SP_CacheAutoLoaderMulti": SP_CacheAutoLoaderMulti,
   "SP_CacheAutoLoaderMulti10": SP_CacheAutoLoaderMulti10,
   "SP_CacheRouter": SP_CacheRouter,
   "SP_CacheDirToKeys": SP_DirToKeys,
   "SP_CacheCleanKeyString": SP_CleanKeyString,
   "SP_CacheDataPacker": SP_DataPacker,
   "SP_CacheDataUnpacker": SP_DataUnpacker,
}