import re
import requests

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
                "delete_loras": (["False", "True"], {"default": "False"}),
                "page": ("INT", {"default": 1, "min": 1, "max": 1000}),
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

    def doit(self, limit, nsfw, sort, period, delete_loras, page, **kwargs):
        if not nsfw:
            nsfw = None

        params = {
            "limit": limit, # 0 - 200
            "nsfw": nsfw, # (None, Soft, Mature, X)
            "sort": sort, # (Most Reactions, Most Comments, Newest)
            "period": period, # (AllTime, Year, Month, Week, Day)
            "page": page
        }

        response = requests.get("https://civitai.com/api/v1/images", params=params)

        prompts = []
        if response.status_code == 200:
            data = response.json()
            
            for i in data['items']:
                meta = i.get('meta', None)
                if not meta:
                    continue

                prompt = meta.get('prompt', None)
                if not prompt:
                    continue
                
                prompt = prompt.replace('\r', '').replace('\n', ' ')

                prompts.append(re.sub(r'<[^>]*>', '', prompt) if delete_loras == 'True' else prompt)
        else:
            print("Error:", response.status_code)

        return '\n'.join(prompts), 

NODE_CLASS_MAPPINGS = {
    "CivitaiPrompts": CivitaiPrompts,
}


'''
"items":[
      {
         "id":234845,
         "url":"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/cc00cfce-393c-476e-c71f-562722b77c00/width=512/cc00cfce-393c-476e-c71f-562722b77c00.jpeg",
         "hash":"UFK,,857O?-o1iS~Ejj@~Wxu}?IASh9F4nod",
         "width":512,
         "height":1024,
         "nsfwLevel":"Mature",
         "nsfw":true,
         "createdAt":"2023-03-12T06:29:56.461Z",
         "postId":116584,
         "stats":{
            "cryCount":19,
            "laughCount":41,
            "likeCount":655,
            "dislikeCount":14,
            "heartCount":1037,
            "commentCount":6
         },
         "meta":{
            "ENSD":"31337",
            "Size":"512x1024",
            "seed":3984980528,
            "Model":"chilloutmix_NiPrunedFp32Fix",
            "steps":25,
            "prompt":"<lora:virtualgirlRin_v30:0.3>, <lora:upshirtUnderboob_v10:1.35>,\n(RAW photo:1.2),(photorealistic:1.4),(masterpiece:1.3),(best quality:1.4),ultra high res, HDR,8k resolution,\ndreamlike, check commentary, commentary request, scenery,((no text)),\n1girl,  (cleavage:1.5), (large breasts:1.5), (pubic hair:1.5),(lifting shirt), (detailed laced underpants:1.4), (full body), (looking down:1.5), (close up), look at the viewer, naughty face, (touching self hair:1.3), (tattoo:1.3), topless, arm, (close up:1.5), (focus on breasts),\n(detailed eyes),(detailed facial features), (detailed clothes features), (breast blush)\n\ntrending on cg society, plasticine,  bob hair, (strong and toned abs), ((beautiful woman)), wearing  choker, thigh choker, very pale white skin, in snow, feminine and muscular, smiling, wet skin, female focus,",
            "sampler":"DPM++ SDE Karras",
            "cfgScale":8,
            "Clip skip":"2",
            "resources":[
               {
                  "name":"virtualgirlRin_v30",
                  "type":"lora",
                  "weight":0.3
               },
               {
                  "name":"upshirtUnderboob_v10",
                  "type":"lora",
                  "weight":1.35
               },
               {
                  "hash":"fc2511737a",
                  "name":"chilloutmix_NiPrunedFp32Fix",
                  "type":"model"
               }
            ],
            "Model hash":"fc2511737a",
            "negativePrompt":"EasyNegative,bad_prompt,ng_deepnegative_v1_75t,(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), bad hands, missing fingers, extra digit, (futa:1.1), bad body, glans,",
            "Face restoration":"GFPGAN"
         },
         "username":"noevils"
      }
'''