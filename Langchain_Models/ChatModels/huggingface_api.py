from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1",
    provider="fireworks-ai",
    task="conversational",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    max_new_tokens=128,
    temperature=0.0,
    do_sample=False,
)

chat_llm = ChatHuggingFace(llm=endpoint)

# 3. Call it as a flat string prompt
response = chat_llm.invoke("How to lose weight")

print(response)
