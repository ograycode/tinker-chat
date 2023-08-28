from langchain.callbacks.manager import Callbacks
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.chains.transform import TransformChain
from typing import Any, Dict, List
from src.models import Route

class WeatherRetriever(BaseRetriever):
    url: str
    def get_relevant_documents(self, query: str, *, callbacks: Callbacks = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> List[Document]:
        headers = {
            'User-Agent': 'python-requests/2.31.0',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
        loader = WebBaseLoader(self.url, header_template=headers)
        return loader.load()
    
def weather():
    return Route(
        "weather",
        "Good for answering questions about the weather, tempature and forcasts",
        TransformChain(input_variables=["query"], output_variables=["text"], transform=lambda q: {"text": WeatherRetriever().get_relevant_documents(q)[0].page_content})
    )