import shutil
from os import path
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.schema import Document

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_and_load_index(data_dir="./data", persist_dir="./data") -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    dir_loader = DirectoryLoader(data_dir, show_progress=True, loader_cls=UnstructuredFileLoader)
    docs = dir_loader.load_and_split(CharacterTextSplitter(chunk_size=500, chunk_overlap=10))
    return Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

def load_index(persist_dir="./data/.index") -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)

def load_or_create(data_dir="./data", persist_dir="./data/.index", refresh=False) -> Chroma:
    persist_dir_exists = path.exists(persist_dir)
    if persist_dir_exists and not refresh:
        return load_index(persist_dir=persist_dir)
    elif persist_dir_exists and refresh:
        shutil.rmtree(persist_dir)
    return create_and_load_index(data_dir=data_dir, persist_dir=persist_dir)

def in_memory(docs: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    split_txt = CharacterTextSplitter(chunk_size=500, chunk_overlap=10).split_documents(docs)
    return Chroma.from_documents(split_txt, embeddings)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)