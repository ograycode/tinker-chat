import json
from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseSettings, BaseModel
from langchain.chains.base import Chain
from langchain.chains import RetrievalQA
from src.vectore_store import Coordinator
from src.llm import LLM, get_llm

CONFIG_FILE_NAME: str = "config.json"

def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    A simple settings source that loads variables from a JSON file
    at the project's root.

    Here we happen to choose to use the `env_file_encoding` from Config
    when reading `config.json`
    """
    encoding = settings.__config__.env_file_encoding
    file = Path(CONFIG_FILE_NAME)
    if file.exists():
        return json.loads(file.read_text(encoding))
    return {}


class Route(BaseModel):
    name: str
    description: str
    chain: Chain


class RAG(BaseModel):
    name: str
    description: str
    directory: str
    persist_directory: str
    llm: LLM = LLM.ORCA_MINI
    
    def create_route(self) -> Route:
        """
        Creates a Route instance with the given LangChain Language Model (llm).
    
        Returns:
            A Route instance.
    
        Raises:
            FileNotFoundError: If the specified index does not exist.
        """
        llm = get_llm(self.llm)
        docsearch = Coordinator().load_or_create(self.directory, self.persist_directory).as_retriever()
        return Route(
            name=self.name,
            description=self.description,
            chain=RetrievalQA.from_llm(llm, retriever=docsearch, output_key="text")
        )


class AppSettings(BaseSettings):
    vector_storage_subdir: str = ".index"
    rags: List[RAG] = []
    default_llm: LLM = LLM.ORCA_MINI
    
    def save(self):
        with open(CONFIG_FILE_NAME, "w") as f:
            json.dump(self.dict(), f)
    
    class Config:
        env_file_encoding = 'utf-8'

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                json_config_settings_source,
                env_settings,
                file_secret_settings,
            )
