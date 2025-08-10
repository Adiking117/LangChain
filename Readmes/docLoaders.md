Document Loaders
================
**RAG**
Technique which combines info retrieval with language generation, where a model retrieves relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses.
Benefits of RAG :
    1.use of up-to-date information
    2.better privacy
    3.no limit of doc size

->100s of doc loaders - most impt concept/principal and 4 most used doc loaders

![alt text](documentLoaders\Rag.jpg)

**Document loaders in langchain**
![alt text](documentLoaders\docLoaders.jpg)

-components in langchain used to load data from various sources into a standardized format(usually as doc objects) which can be then used for chunking,embedding,retrieval and generation.
*Document(
    pageContent ="the actual content text",
    metadata ={"source":"finame.pdf", ...}
)*
-a python list

**TextLoader**
.txt plain text (only txt file)-> Document objects
-Ideal for loading chat logs, scraped text, transcripts, code,snippets, any plain text. in to langchain pipeline
