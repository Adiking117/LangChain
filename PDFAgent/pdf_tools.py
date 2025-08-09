# pdf_tools.py
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from config import EMBED_MODEL, PERSIST_DIR, CHROMA_COLLECTION_NAME
import os
from langchain_huggingface import HuggingFaceEmbeddings
os.environ['HF_HOME'] = 'C:/Users/Aditya/Desktop/Langchain/Langchain_Models/LOCALINSTALLEDMODELS'
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


def load_pdf_to_docs(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load PDF and split into chunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = splitter.split_documents(pages)
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("chunk", i)
    return docs

def create_or_load_vectorstore(docs: list[Document] = None):
    """Create or load persisted Chroma vector DB."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings
        )
        if docs:
            vectordb.add_documents(docs)
            vectordb.persist()
    else:
        vectordb = Chroma.from_documents(
            docs, embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
        vectordb.persist()

    return vectordb
