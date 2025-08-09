# agent_tools.py
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from config import LLM_MODEL
from pdf_tools import load_pdf_to_docs, create_or_load_vectorstore
from retriever_tools import SimpleChatWrapper, CompressedRetriever

class PDFAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.simple_llm = SimpleChatWrapper(LLM_MODEL, temperature=0)
        self.vectordb = None
        self.retriever = None

    def load_pdf(self, pdf_path: str):
        docs = load_pdf_to_docs(pdf_path)
        self.vectordb = create_or_load_vectorstore(docs)
        base_retriever = self.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20}
        )
        self.retriever = CompressedRetriever(base_retriever, self.simple_llm)
        return f"PDF loaded with {len(docs)} chunks."

    def summarize(self, query: str):
        if not self.retriever:
            return "No document loaded."
        docs = self.retriever.get_relevant_documents(query)
        context = "\n---\n".join([d.page_content for d in docs])
        prompt = f"Summarize the following:\n{context}\n\nSummary:"
        return self.simple_llm.call_as_llm(prompt)

    def answer_question(self, query: str):
        if not self.retriever:
            return "No document loaded."
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.retriever
        )
        return qa.run(query)

    def generate_quiz(self, topic: str = None):
        """Generate a 10-question quiz from the loaded PDF."""
        if not self.retriever:
            return "No document loaded."

        # Use the whole document's most relevant chunks
        query = topic if topic else "Create a quiz from this document"
        docs = self.retriever.get_relevant_documents(query)
        context = "\n---\n".join([d.page_content for d in docs])

        prompt = (
            f"You are a quiz generator. Based on the following content:\n{context}\n\n"
            "Create 10 quiz questions that test understanding of the material. "
            "Mix multiple choice (with 4 options and correct answer marked) "
            "and short answer questions. Ensure variety and clarity.\n\n"
            "Format:\nQ1: ...\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: ...\n\n"
            "or for short answer:\nQx: ...\nAnswer: ..."
        )
        return self.simple_llm.call_as_llm(prompt)


def build_react_agent():
    agent_instance = PDFAgent()

    tools = [
        Tool(
            name="LoadPDF",
            func=agent_instance.load_pdf,
            description="Load a PDF from path into memory."
        ),
        Tool(
            name="SummarizePDF",
            func=agent_instance.summarize,
            description="Summarize the loaded PDF."
        ),
        Tool(
            name="AskPDF",
            func=agent_instance.answer_question,
            description="Ask a question about the loaded PDF."
        ),
        Tool(
            name="QuizGenerator",
            func=agent_instance.generate_quiz,
            description="Generate a 10-question quiz from the loaded PDF."
        )
    ]

    llm_for_agent = ChatOpenAI(model=LLM_MODEL, temperature=0)
    agent = initialize_agent(
        tools,
        llm_for_agent,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )

    return agent, agent_instance
