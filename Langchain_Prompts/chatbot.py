from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOpenAI()

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
