from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import sys
sys.path.append('C:/Users/Aditya/Desktop/Langchain/')
from Langchain_Models.ChatModels.hugface_tiny_llama import model


# # 1. Without context chatbot
# while True:
#     user_input = input('You: ')
#     if user_input == 'exit':
#         break
#     result = model.invoke(user_input)
#     print("AI: ",result.content)


# # 2. With context - Problem : Cant understand whose message is whos? Solved in 3
# chat_history = []

# while True:
#     user_input = input('You: ')
#     chat_history.append(user_input)
#     if user_input == 'exit':
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(result.content)
#     print("AI: ",result.content)

# print(chat_history)


# 3. Messages
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input)) # type: ignore
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) # type: ignore
    print("AI: ",result.content)

print(chat_history)