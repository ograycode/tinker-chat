from langchain.schema.language_model import BaseLanguageModel
from src.fastchain import FastChain
from src.llm import get_llm
from src.routes import default_route, wikipedia
from src.models import AppSettings


def create_ai(llm: BaseLanguageModel):
    ai = FastChain()
    settings = AppSettings()

    for rag in settings.rags:
        ai.add_route(rag.create_route(llm))
    ai.add_route(wikipedia(llm))
    route = default_route(llm)
    ai.add_route(route)
    ai.add_route(route, default_route=True)
    return ai