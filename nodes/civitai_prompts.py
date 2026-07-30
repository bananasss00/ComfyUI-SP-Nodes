import re
import requests
import logging

# Настройка логирования
logger = logging.getLogger("CivitaiPrompts")
logger.setLevel(logging.INFO)

class CivitaiPrompts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        inputs = {
            "required": {
                "limit": ("INT", {"default": 200, "min": 0, "max": 200}),
                "nsfw": (["", "Soft", "Mature", "X"], {"default": "Soft"}),
                "sort": (["Most Reactions", "Most Comments", "Newest"], {"default": "Most Reactions"}),
                "period": (["AllTime", "Year", "Month", "Week", "Day"], {"default": "Week"}),
                "delete_loras": ("BOOLEAN", {"default": True}),
                "pages": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
            "optional": {

            },
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "doit"
    # OUTPUT_NODE = True

    CATEGORY = 'SP-Nodes'

    def parse_page(self, limit, nsfw, sort, period, delete_loras, cursor):
        if not nsfw:
            nsfw = None

        params = {
            "limit": limit,
            "nsfw": nsfw,
            "sort": sort,
            "period": period,
            "withMeta": True, # Требуется для принудительного включения метаданных генерации
        }
        
        if cursor:
            params["cursor"] = cursor

        # Эмуляция браузера для обхода базовых проверок Cloudflare
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        logger.info(f"Sending GET request to Civitai API with params: {params}")

        try:
            response = requests.get(
                "https://civitai.com/api/v1/images", 
                params=params, 
                headers=headers, 
                timeout=15
            )
            logger.info(f"Request URL: {response.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Civitai API: {str(e)}", exc_info=True)
            return [], None

        prompts = []
        next_cursor = None

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {str(e)}. Response body starts with: {response.text[:300]}")
                return [], None
            
            metadata = data.get('metadata', {})
            if metadata:
                next_cursor = metadata.get('nextCursor', None)
                logger.info(f"Found next page cursor: {next_cursor}")
            else:
                logger.warning("No metadata block found in the API response.")

            items = data.get('items', [])
            logger.info(f"Received {len(items)} items from the page.")

            meta_count = 0
            prompt_count = 0

            for i in items:
                meta = i.get('meta', None)
                if not meta:
                    continue
                meta_count += 1

                prompt = meta.get('prompt', None)
                if not prompt:
                    continue
                prompt_count += 1
                
                prompt = prompt.replace('\r', '').replace('\n', ' ')
                prompts.append(re.sub(r'<[^>]*>', '', prompt) if delete_loras else prompt)

            logger.info(f"Processed items: {meta_count} had 'meta' block, {prompt_count} had 'prompt'.")
        else:
            logger.error(f"API Error (Status Code {response.status_code}): {response.text[:500]}")

        return prompts, next_cursor

    def doit(self, limit, nsfw, sort, period, delete_loras, pages, **kwargs):
        logger.info(f"Starting CivitaiPrompts with parameters: limit={limit}, nsfw={nsfw}, sort='{sort}', period='{period}', delete_loras={delete_loras}, pages={pages}")
        prompts = []
        cursor = None

        for page_idx in range(pages):
            logger.info(f"Fetching page {page_idx + 1}/{pages}...")
            page_prompts, next_cursor = self.parse_page(limit, nsfw, sort, period, delete_loras, cursor)
            prompts.extend(page_prompts)
            
            if not next_cursor:
                logger.info("No next cursor returned. Stopping pagination.")
                break
            cursor = next_cursor

        logger.info(f"Finished processing. Total prompts successfully collected: {len(prompts)}")
        return '\n'.join(prompts), 

NODE_CLASS_MAPPINGS = {
    "CivitaiPrompts": CivitaiPrompts,
}