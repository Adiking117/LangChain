# retriever_tools.py
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import os
from langchain_huggingface import HuggingFaceEmbeddings
os.environ['HF_HOME'] = 'C:/Users/Aditya/Desktop/Langchain/Langchain_Models/LOCALINSTALLEDMODELS'
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


class SimpleChatWrapper:
    """Minimal wrapper to call ChatOpenAI like a normal LLM."""
    def __init__(self, model_name, temperature=0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def call_as_llm(self, prompt: str) -> str:
        resp = self.llm.generate([{"role": "user", "content": prompt}])
        return "\n".join([gen[0].text for gen in resp.generations])

class CompressedRetriever:
    """Retriever with MMR + contextual compression."""
    def __init__(self, base_retriever, llm_wrapper, compress_prompt=None):
        self.base_retriever = base_retriever
        self.llm = llm_wrapper
        self.compress_prompt = compress_prompt or (
            "Extract the most relevant sentences from the text for answering the question.\n"
            "Question: {question}\nText: {text}\n\nAnswer:"
        )

    def get_relevant_documents(self, query: str, k: int = 4):
        docs = self.base_retriever.get_relevant_documents(query)
        compressed_docs = []

        for doc in docs:
            prompt = self.compress_prompt.format(question=query, text=doc.page_content)
            compressed_text = self.llm.call_as_llm(prompt)
            compressed_docs.append(Document(page_content=compressed_text, metadata=doc.metadata))

        return compressed_docs
