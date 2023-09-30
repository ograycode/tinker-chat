import shutil
import copy
from os import path
from typing import Callable, Dict, List, Optional, Sequence
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter, Language
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.schema import Document
from langchain.schema.vectorstore import VectorStore


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Dictionary mapping file extensions to language, class, chunk_size, and chunk_overlap
SPLITTER_MAPPING = {
    ".py": {
        "args": (Language.PYTHON,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".cpp": {
        "args": (Language.CPP,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".go": {
        "args": (Language.GO,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".java": {
        "args": (Language.JAVA,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".js": {
        "args": (Language.JS,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".php": {
        "args": (Language.PHP,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".proto": {
        "args": (Language.PROTO,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".rst": {
        "args": (Language.RST,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".ruby": {
        "args": (Language.RUBY,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".rust": {
        "args": (Language.RUST,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".scala": {
        "args": (Language.SCALA,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".swift": {
        "args": (Language.SWIFT,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".md": {
        "args": (Language.MARKDOWN,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".tex": {
        "args": (Language.LATEX,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".html": {
        "args": (Language.HTML,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".sol": {
        "args": (Language.SOL,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    ".cs": {
        "args": (Language.CSHARP,),
        "callable": RecursiveCharacterTextSplitter.from_language,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    },
    "default": {
        "args": set(),
        "callable": RecursiveCharacterTextSplitter,
        "kwargs": {"chunk_size": 500, "chunk_overlap": 10}
    }
}


class Coordinator:

    def __init__(self,
                 vector_store: VectorStore = Chroma,
                 embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
                 splitter_mapping=SPLITTER_MAPPING) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.splitter_mapping = splitter_mapping

    def get_splitter(self, doc: Document) -> TextSplitter:
        source_file: str = doc.metadata.get("source", "")
        file_extension = source_file[source_file.rfind("."):]
        mapped = self.splitter_mapping.get(file_extension, self.splitter_mapping["default"])
        return mapped["callable"](*mapped["args"], **mapped["kwargs"])

    
    def get_loader(self, data_dir):
        return DirectoryLoader(data_dir,
                               show_progress=True,
                               loader_cls=UnstructuredFileLoader,
                               silent_errors=True)
    

    def create_and_load_index(self, data_dir, persist_dir) -> VectorStore:
        dir_loader = self.get_loader(data_dir)
        split_docs: Sequence[Document] = []
        docs = dir_loader.load()
        for doc in docs:
            split_docs.extend(self.get_splitter(doc).split_documents([doc]))
        return self.vector_store.from_documents(split_docs, self.embeddings, persist_directory=persist_dir)


    def load_index(self, persist_dir) -> VectorStore:
        return self.vector_store(embedding_function=self.embeddings, persist_directory=persist_dir)


    def load_or_create(self, data_dir, persist_dir, refresh=False) -> Chroma:
        persist_dir_exists = path.exists(persist_dir)
        if persist_dir_exists and not refresh:
            return self.load_index(persist_dir)
        elif persist_dir_exists and refresh:
            shutil.rmtree(persist_dir)
        return self.create_and_load_index(data_dir, persist_dir)


    def in_memory(self, docs: List[Document]):
        split_docs = self.get_splitter(docs)
        return self.vector_store.from_documents(split_docs, self.embeddings)


    def destroy(self, persist_dir):
        persist_dir_exists = path.exists(persist_dir)
        if persist_dir_exists:
            shutil.rmtree(persist_dir)