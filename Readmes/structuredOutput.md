Structured Output
=================

![structuredOutput drawio](https://github.com/user-attachments/assets/8a0dd1bf-8501-4ef4-a457-3223bcc9f068)
![strOP2](https://github.com/user-attachments/assets/7b33b342-f195-4755-9940-daf70f62af8a)
![strdOP3](https://github.com/user-attachments/assets/a478f67e-44ec-42a0-8a3b-15ab997c5ab0)






# Need ?
-Data Extraction ( eg. nukri.com>resume uploaded> Extract all text like name,college,10th,12th etc. store into database > send to llm>get data in json format>store in DB via code )
- API building (eg. Amazon Review >via API  extract all review topics for diff. products,overall sentiment etc )
- Agents [chatbots on steroids] (eg. calculator agent > prompt-tell sqrt of 2 - text cant be given to calc. as its expecting numbers -  run str output on prompt and fetch numbers - 2 and sqrt - will give output)

# Ways to get Structured Output

*with_structured_output*: before calling model_invoke() we call this function and give the data format > data format specify 2/3 ways : Typed Dict; Pydantic; JSON Schema
1.) ***Typed Dictionary*** : In python we just create onthe go dict. > person = {"name" : "aman", "age" : 35}, Issue : suppose 2 progrmrs working on same projct /file and he treats age as string instead of no. so issue in runtime.
<ins>Solun : define first how the dict will look like > create class  person and in that dict name key , age key : will have string and int as values respectively. > tells Py what keys are required & what types of values they should have</ins>
<ins>Drawback :  If let age is int even then we can keep age value as string.>code will run > NO Validation</ins>
               

***Problem Statement*** > Reviews of phone > send to llm > o/p should be structured (dict).keys>summary and sentiment 
Code ? working > when we call with_structures_output() and provides a structure/schema -behind the scene a prompt gets generated >then a review gets attached > llm is trained to give json output > so gives the same.

-**Annotated** : LLM is well trained> got to know whats summary and sentiment, but there can be ambuigity sometime to solve this we can guide llm by attacking line so that llm can get to know. this is known as annotation.*

2.) ***Pydantic***
Data validation and data parsing library for py. Ensures the data is type safe, correct, structured.



# json schema
-we use json schema when our project is in multiple languages 
    -like FE(jSCRIPT) BE(python) json schema needed in both so no pydantic or typedisct, use json
    *eg:* 
    {
    "title":"student",
    "description":"schema on stud",
    "type":"obj",
    "properties":{
        "name":"string",
        "age":"integer"
    },
    "required":["name"]
}

# When_To_Use_What?
**Typedict** : Use python in whole project, only need type hints, no data validation, Trust llm to return correct data
**Pydantic** : Need data validation, need to ensure data is correct, need default values, Automatic type conversion
**Json Schema** : Project is multi-language, need validation but dont need python objs, structure in standard json format.