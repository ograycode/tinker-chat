from src.fastchain import FastChain
from src.llm import get_llm
from src.routes import default_route
from src.models import AppSettings

def create_ai():
    ai = FastChain()
    settings = AppSettings()
    
    llm = get_llm()

    for rag in settings.rags:
        ai.add_route(rag.create_route(llm))
    route = default_route(llm)
    ai.add_route(route)
    ai.add_route(route, default_route=True)
    return ai