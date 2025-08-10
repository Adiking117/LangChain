# Output Parsers
-In langchain it helps convert raw output from llm's to structured format like JSON, csv, pydantic objects etc. Ensures consistency validation and ease of use.
-can and can't (with_structured_outptuy) with both the models we can use output parsers.

***Types***
-Many types used, but 4 most used->
1>String    2>JSON  3>Structured    4>Pydantic

**StrOutput Parser**
-Simplest Output Parser in langchain. Parse the output of  LLM nad return it as plain string.
    -Usecase:
        Topic > LLM > Detailed Report > LLM > 5 lines summarize

**JSON OutputParser**
-Converts raw output to JSON format. : llm > json output
-Whenever we use any outputparser , during prompt formation we send additional instruction (what kind of output we need from LLM)-format_instruction > parser tells this when we call *get_format_instructions()*
-format_instruction is partial variable because it fills before runtime(user not giving)-with help of func call
-Problem : cant enforce schema(what structure json output have)

**Structured_Output_Parser**
-Extract structured JSON data from LLM responses based on predefined *field schemas*
-defines a list of fields (ResponseSchema) that the model should return, ensuring the output follows a structured format. 
Disadvantage :
-data validation cant be done, suppose in schema we wanted name, age, city of a person, should be string, integer, string respectively. But got 35 years in response >its a string. can write in prompt but not sure . 

**Pydantic_output_parser**
-uses pydantic models to enforce schema validation when processing LLM responses.
-*Strict Schema Enforcement*:ensure LLM response follow a well-structured schema 
-*type safety* : automatically convert llm output to python objcts
-*easy validation* : pydantic built-in validn to catch incorrect/missing data
-*seamless integration* : works well with other langchain components


