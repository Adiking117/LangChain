from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import sys
sys.path.append('C:/Users/Aditya/Desktop/Langchain/')
from Langchain_Models.ChatModels.hugface_tiny_llama import model

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
