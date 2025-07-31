from langchain_core.prompts import PromptTemplate
import sys
sys.path.append('C:/Users/Aditya/Desktop/Langchain/')
from Langchain_Models.ChatModels.hugface_tiny_llama import model

template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

prompt = template2.invoke({'name':'Adi'})

print(prompt)

result = model.invoke(prompt)

print(result.content)
