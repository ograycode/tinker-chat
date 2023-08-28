from langchain.llms import Ollama
from langchain.schema.language_model import BaseLanguageModel

def get_llm() -> BaseLanguageModel:
    return Ollama(base_url="http://localhost:11434",
                 model="orca-mini",)