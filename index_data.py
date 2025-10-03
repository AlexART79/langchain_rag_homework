import os.path
from typing import List

from chromadb import Documents
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader


base_path = os.path.dirname(os.path.abspath(__file__))
chroma_db = os.environ.get("CHROMA_DB_PATH", "chroma_db")

chroma_db_path = os.path.join(base_path, chroma_db)
data_files_path = "./data"

embedding_func = OllamaEmbeddings(model="nomic-embed-text")

def load_files(path: str) -> List[Documents]:

    # list files in dir
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # load files
    docs = []
    for f_path in files:
        file_path = os.path.join(path, f_path)
        loader = PyPDFLoader(file_path=file_path)
        docs_partial = loader.load()
        docs.extend(docs_partial)

    return docs

if not os.path.exists(chroma_db_path):
    print("Creating vector store at", chroma_db_path)

    docs = load_files(data_files_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_func,
        collection_name="rag-chroma",
        persist_directory=chroma_db_path
    )

print("Loading vector store from", chroma_db_path)
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=chroma_db_path,
    embedding_function=embedding_func,
).as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})