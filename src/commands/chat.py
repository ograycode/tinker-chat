from src.llm import get_llm
from src.models import AppSettings
from src.routes import default_route, wikipedia
from src.fastchain import FastChain


def main():
    settings = AppSettings()
    ai = FastChain(settings)
    llm = get_llm()
    ai.add_route(default_route(llm), default_route=True)
    ai.add_route(wikipedia(llm))
    ai.serve()
