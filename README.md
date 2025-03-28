# Multi-Modality
This is a REACT(Reasoning and Action Agent) based RAG(Retrieval Augumented Generation) Application powered by OpenAI.

Work Flow:
1. The user is provided file upload option and a text box.
2. User may or may not upload a file.
3. User asks a query.
4. The Agent will decide how to handle the query and for this it has options to use different tools such as:
   a. VectorDB tool: To search for best responses from vectorstore, this will only be possible if agent finds a file uploaded by user.
   b. LLM tool: The agent can generate a response using LLM.
   c. Web Search tool: in case answer could not be fetched from vector store or LLM, then Agent can get answer Online using SERP API.
   d. Image tool: if the file uploaded is of type jpg,png then the the agent uses gpt4o to ananlyze the image and generate a response.

Tech Stack:
1. langchain
2. OpenaAI
3. Python
4. HuggingFace
5. SERP API
6. Streamlit
