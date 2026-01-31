import logging
import os
import sys
import joblib
from comfy.comfy_types import IO
import numpy as np
import torch
import server
from aiohttp import web

# Set up a logger for the nodes
logger = logging.getLogger(__name__)

# A global dictionary to serve as an in-memory cache for all nodes in this file.
CACHE = {}

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
    ACTION_LIST = ["view", "clear_key", "clear_all"]

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
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "If False, the node will simply pass the value through without any caching."}),
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

    def check_lazy_status(self, key, cache_directory, enabled, value=None):
        """
        Correctly determines if the lazy input 'value' needs to be evaluated.
        """
        if value is not None:
            return []

        if not enabled:
            return ["value"]

        filepath = self._get_filepath(cache_directory, key)
        if key not in CACHE and not os.path.exists(filepath):
            return ["value"]

        return []

    def load_or_compute(self, key, cache_directory, enabled, value=None):
        """
        Loads data from cache (memory/disk) or computes it if not found.
        """
        if not enabled:
            if value is None:
                raise ValueError(f"Caching is disabled for key '{key}', but no input was provided to 'value'.")
            return ([value], "Cache disabled; value passed through",)

        if key in CACHE:
            return ([CACHE[key]], "Loaded from memory",)

        filepath = self._get_filepath(cache_directory, key)
        if os.path.exists(filepath):
            try:
                loaded_value = joblib.load(filepath)
                CACHE[key] = loaded_value
                return ([loaded_value], "Loaded from disk",)
            except Exception as e:
                logger.error(f"Failed to load cache file {filepath}, will re-compute. Error: {e}")

        if value is None:
            raise ValueError(f"Cache MISS for key '{key}' in memory and on disk, but no input was provided to 'value'.")

        CACHE[key] = value

        try:
            os.makedirs(cache_directory, exist_ok=True)
            joblib.dump(value, filepath)
        except Exception as e:
            logger.error(f"Failed to save cache file to {filepath}: {e}")

        return ([value], "Newly computed and cached",)

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


NODE_CLASS_MAPPINGS = {
   "SP_CacheValue": SP_CacheValue,
   "SP_CacheManager": SP_CacheManager,
   "SP_CacheCheck": SP_CacheCheck,
   "SP_CacheStore": SP_CacheStore,
   "SP_CacheGet": SP_CacheGet,
   "SP_CompositeCacheKey": SP_CompositeCacheKey,
   "SP_CachePersistence": SP_CachePersistence,
   "SP_CacheAutoLoader": SP_CacheAutoLoader,
}