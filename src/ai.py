from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from src.fastchain import FastChain, Route
from src.llm import get_llm
from src.models import AppSettings

def create_ai():
    ai = FastChain()
    settings = AppSettings()
    
    llm = get_llm()

    for rag in settings.rags:
        ai.add_route(rag.create_route(llm))
    route = Route(
        name="everything_else",
        description="A comprehensive, versatile chatbot designed for multi-purpose interactions, adept at handling a broad spectrum of queries, topics, and discussions.",
        chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template("You are an AI assistant, respond to this in a helpful way: {query}"))
    )
    ai.add_route(route)
    ai.add_route(route, default_route=True)
    return ai