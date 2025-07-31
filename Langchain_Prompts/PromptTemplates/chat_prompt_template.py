from langchain_core.prompts import ChatPromptTemplate
import sys
sys.path.append('C:/Users/Aditya/Desktop/Langchain/')
from Langchain_Models.ChatModels.hugface_tiny_llama import model

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'WWE','topic':'Suplex'})

print(prompt)

result = model.invoke(prompt)

print(result.content)
