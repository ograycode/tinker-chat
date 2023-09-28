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

class Coordinator:
    
    def __init__(self,
                 vector_store: VectorStore = Chroma,
                 embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings
    
    def get_splitter(self, doc: Document) -> TextSplitter:
        if doc.metadata.get("source", "").endswith(".py"):
            return RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=500, chunk_overlap=10)
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    
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


def destroy(persist_dir):
    persist_dir_exists = path.exists(persist_dir)
    if persist_dir_exists:
        shutil.rmtree(persist_dir)