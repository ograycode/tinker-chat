from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from typing import Any, Dict, List
from src.models import Route
from src.llm import get_llm


def default_route(llm: BaseLanguageModel):
    return Route(
        name="everything_else",
        description="A comprehensive, versatile chatbot designed for multi-purpose interactions, adept at handling a broad spectrum of queries, topics, and discussions.",
        chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template("You are an AI assistant, respond to this in a helpful way: {query}"))
    )