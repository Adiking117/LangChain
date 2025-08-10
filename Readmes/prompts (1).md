-*Prompts are the input/instructions/queries given to model to guide its output*
**Types**
1) Text based
2) Multimodal (images/videos/sound)

**STATIC Prompt**- User gives different input/prompt each time to get a new response from llm- not used much-bcs when user is able to write static prompt -user is getting more control. Outputs depend on prompt too much(sensitive).More chance of mistakes if we take static prompt from user
-*user should get consistent experience*

**Solution**-*use/prepare template-**dynamic prompt**

**Prompt Template**
-Structured way to create prompts dynamically by inserting variables into a predefined template
-No hardcoding of prompts
-Define placeholders that can be filled at runtime with different inputs.
-*Resuable, flexible, easy to manage, -dunamic user inputs / automated workflows*

**Q. wHY USE PROMPTTEMPLATE OVER FSTRING?**
-DEFAULT validation-*validate_template=True*
-Resuable - can use template as a json file (when template is required for multiple pages)-prompt_generator>saved as template.json
-Langchain ecosystem 

#Template.json file automatically created when after running prompt_generator.py file
#invoke is called 2 time - for prompt template and for model in response.- 2 step process>prompt designed then sending to llm 
- **use CHAIN** to call invoke only once >tie both steps to create chain template and model

prompt_in_langchain
![invoke](https://github.com/user-attachments/assets/d883b073-46df-4437-98ee-be6fdbbf8633)

<img width="368" alt="image" src="https://github.com/user-attachments/assets/85ef73b9-6580-49d6-b0c0-64bdb0ea1148" /> 
wrong : <img width="758" alt="image-1" src="https://github.com/user-attachments/assets/09cbd7b3-548d-4e99-92b8-f6ed11c227e1" />


-*LangChain's ChatPromptTemplate.from_messages() expects a list of tuples (("role", "message")), not actual SystemMessage or HumanMessage objects.*


**Message Placeholder**
-in langchain is a special placeholder used inside a ChatPromptTemplate to dynamically insert chat history or a list of messages at runtime.
-eg. A customer raised a return related issue for his order, after few days again he asks for refund , chat_history.txt is a file somewhere stored in cloud db for eg. .
=======
**Code Snippets and modifications**
-Code without chat history passed to model
<img width="256" alt="image" src="https://github.com/user-attachments/assets/bae8b55d-f582-42dd-ab01-6264bc72cbd0" />
-Output > <img width="727" alt="image" src="https://github.com/user-attachments/assets/49594031-0ba1-418c-814e-3c9ddf365f8f" />
-Output after using chat_history: <img width="722" alt="image" src="https://github.com/user-attachments/assets/0592c5e9-3bd9-43e7-bbe7-e55bff384041" />
-*All messages as it is who sent what no information-will be difficult for llm if particular message was by user or llm-create error*
-#therefore when maintaining a caht history>store all the messages awa who sent that message> maintain a dictionary user and a#
-*NO NEED OF MAINTAINING DICTIONARY > SOLVED BY LANGCHAIN LIBRARY -BUIT IN CLASSES*

**#Messages in Langchain**(refer messages.py file)
1.) System messages
2.) Human Message
3.) AI Message
