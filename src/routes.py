from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.retrievers import WikipediaRetriever
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from typing import Any, Dict, List
from src.models import Route
from src.vectore_store import in_memory
from src.llm import get_llm


def default_route(llm: BaseLanguageModel):
    return Route(
        name="everything_else",
        description="A comprehensive, versatile chatbot designed for multi-purpose interactions, adept at handling a broad spectrum of queries, topics, and discussions.",
        chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template("You are an AI assistant, respond to this in a helpful way: {query}"))
    )


def wikipedia(llm: BaseLanguageModel):
    docsearch = WikipediaVectorRetriever()
    return Route(
        name="wikipedia",
        description="wikipedia search",
        chain=RetrievalQA.from_llm(llm, retriever=docsearch, output_key="text")
    )


class WikipediaVectorRetriever(WikipediaRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = super()._get_relevant_documents(query, run_manager=run_manager)
        retriever = in_memory(docs).as_retriever()
        return retriever.get_relevant_documents(query, callbacks=run_manager)