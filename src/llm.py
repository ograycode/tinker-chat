from enum import StrEnum
from functools import lru_cache
from langchain.llms import Ollama, FakeListLLM
from langchain.schema.language_model import BaseLanguageModel

class LLM(StrEnum):
    ORCA_MINI = "orca-mini"
    FAKE = "fake"

def get_llm(llm_key: LLM = LLM.ORCA_MINI, use_cache=True) -> BaseLanguageModel:
    if use_cache:
        return cached_get_llm(llm_key)
    else:
        return uncached_get_llm(llm_key)


@lru_cache()
def cached_get_llm(llm_key: LLM) -> BaseLanguageModel:
    return uncached_get_llm(llm_key)


def uncached_get_llm(llm_key: LLM):
    if llm_key == LLM.ORCA_MINI: pass
    elif llm_key == FakeListLLM: return FakeListLLM(responses=["fake"])
    return Ollama(base_url="http://localhost:11434",
                model="orca-mini",)