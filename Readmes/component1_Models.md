**Models**: -Core interfaces through which we interact with AI models.

**Problem**: *Different llm api providers >different codes> suppose i have to use 2 different models in my app so have to write different codes eg openAI and Claude 
         both will have diferent code.And responses also different of different APIs.*
**Solution**:Langchain > *an interface through which we can use any LLM API in standardized fashion. Not much changes in code.*

Two types of AI models for communicate in langchain:

![langchainModels](https://github.com/user-attachments/assets/4a18db1a-6d28-481a-95ae-c835491b7275)

**Open Source Models**
-freely available, can be downloaded modified, fine-tuned, and deployed without restrictions from a central provider.
-full control and customization
-can be deployed on premise or cloud
eg: llama, deepseek,mistral,falcon
#Where to find ?- Huggingface(largest open source models repo)
![Alt Text](images/openmodels.png)

Disadvantages
**Commercial/Closed source Models**
- model on companys server, can be interacted via models API > cost of using api.


**LANGUAGE MODELS**
-AI systms designed to process generate and understand natural lang text.
-![languageModels](https://github.com/user-attachments/assets/4c15cbaf-72bf-4f6f-b20a-6fd6a97f59d5)
-LLM : Genaral purpose models - *any NLP Application >text,code,summarizn, ques ans. Text I/P O/P - Old - replaced by chat models.*
-Chat models : conversational task - *I/P :Squence /  Multiple of messages O/P : chat messages - newer models used more comparison to LLMs*
 <img width="314" alt="image" src="https://github.com/user-attachments/assets/05908a43-b768-4c00-b91c-dfbf939d2c95" />

-![openmodels](https://github.com/user-attachments/assets/5b0a7d60-9361-490e-9d7c-941d7d056366)
