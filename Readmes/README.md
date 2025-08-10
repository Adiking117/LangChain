# GenAI
PDF Chat using Langchain. 

![systemDesign](https://github.com/user-attachments/assets/c2d39c16-697f-48f4-b6a1-7653027c5ae0)

System Architecture : 
-User uploads pdf to cloud (AWS S3/ any)
-Load PDF in system using some document loader
-Divide PDFs in chunks (bases on any parameter like pages/chapters/paragraphs)- using any Text Splitter- let 1000 pages
-Embeddings generated for every page- pages sent to embedding model (any model) > embedding/vector generated for each page - 1000 set of vectors
-Embeddings stored in database

-User query for the pdf > sent to same embedding model
-Generating embedding for that query
-1000 vectors in DB this one 1001th query vector compared to all the vectors (distance from vectors) > close distance
-eg. set 5 closest vectors to look for > extract corresponding pages > used in system query
- System Query
- Output >LLM API

**Challenges In developing App**
$1:Brain Build > *How to build/develope something which can understand the query and generate relevant response >**Solution LLMs***
$2:Computation > *Extremely heavy model to be kept on own server/cloud will cause high computation and therefore high cost > **Solution: eg Chatgpt on openai server > can use their API***
$3:System Orchestration > *Moving components like AWS S3,TEXT splitter, embedding, database, llm api and multiple tasks like load pdf, text split, generate embeddings, retrieval from DB all to be executed via a pipeline- complex workflow coding from scratch will cause difficulty - if any llm api or cloud changes made then code again from scratch - **Solution : Langchain***

