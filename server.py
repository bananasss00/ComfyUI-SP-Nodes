from server import PromptServer
from aiohttp import web
from .nodes.group_nodes.Graph import get_missing_nodes

@PromptServer.instance.routes.get("/sp_group_nodes/missing_nodes")
async def missing_nodes(request):
    data = {'data': get_missing_nodes()}
    return web.json_response(data)

def init():
    pass