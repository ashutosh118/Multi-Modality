import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
import base64


serp_api_key="2e2********************************************5"
os.environ["SERPAPI_API_KEY"]=serp_api_key

# gpt 4o
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_ENDPOINT = "https://******************************"
OPENAI_API_KEY = "0a**********************************1"

os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

chat = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    openai_api_version="2023-05-15",
    openai_api_type="azure",
    temperature=0.0,
    verbose="true"
)

model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

def init():
    # Setup Streamlit page
    st.set_page_config(
        page_title="Welcome Mr. Srivastava",
        page_icon="🤖",
        layout="wide"
    )

def get_vectorstore(pdf_docs):
    """
    Create a FAISS vectorstore from PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create vectorstore
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    st.session_state.vectorstore = vectorstore
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Create a Conversational Retrieval Chain.
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def vectorstore_search(conversation_chain,query):
    """
    Search the vectorstore for a relevant answer.
    """
    response = conversation_chain.invoke({'question': query})
    print("vector response=",response)
    return response['answer']

def main():
    init()
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = ""
   
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "image_data_url" not in st.session_state:
        st.session_state.image_data_url = None

    st.header("AshAI-GPT-Agent 🤖")



    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload Your PDF Document")
        pdf_docs = st.file_uploader("Upload PDFs:", accept_multiple_files=True)
        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    st.session_state.vectorstore = get_vectorstore(pdf_docs)
                    print("st.session_state.vectorstore=",st.session_state.vectorstore)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")
        
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Upload an image (JPEG, JPG, PNG)", type=["jpeg", "png", "jpg"])
        if st.button("Process image"):
            if uploaded_image:
                st.write("Image successfully uploaded.")
                with st.spinner("Processing image..."):
                    # Display the uploaded image in the Streamlit app
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                    
                    # Convert the image to a base64 string
                    bytes_data = uploaded_image.read()
                    base64_image = base64.b64encode(bytes_data).decode("utf-8")
                    st.session_state.image_data_url = f"data:image/{uploaded_image.type.split('/')[-1]};base64,{base64_image}"
                    st.info("Image processed successfully!")
            else:
                st.info("Please upload an image file.")

    # Input container (top)
    input_container = st.container()
    with input_container:
        st.markdown("---")  # Divider line
        query = st.text_input(
            "Type your query:",
            value=st.session_state.pending_input,
            placeholder="Enter your message here..."
        )
        query_input = f"{query} use a tool to find an answer"
        print("query_input=",query_input)
        def get_vectorDBresponse(query_input: str):
            if st.session_state.vectorstore:
                convers_chain = get_conversation_chain(st.session_state.vectorstore)
                response_text = vectorstore_search(convers_chain,query_input)
                print("response text vector=",response_text)
                return response_text
        
        def get_gpt4_tool(query_input: str):
            response = chat.invoke(query_input)
            print("response from gpt 4=",response)
            response_text = response.content
            return response_text
        def get_serpresponse(query_input: str):
            search=SerpAPIWrapper()
            response_text=search.run(query_input)
            return  response_text
        def generate_image_description(query_input: str):
            try:
                # Construct the message for the vision model based on user instruction
                msg = chat.invoke([
                    AIMessage(content="You are an expert in reading images and interpreting them."),
                    HumanMessage(content=[
                        {"type": "text", "text": query_input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": st.session_state.image_data_url
                            },
                        }
                    ])
                ])
                # Extract and return the result from the model
                return msg.content if hasattr(msg, "content") else "No result returned."
            except Exception as e:
                return f"An error occurred: {str(e)}"

        vectorDB_tool = Tool.from_function(
            name="VectorDB_tool",
            description="This is the first tool to be used, this tool gives response from the files, if you didn't find any relevant answer then only go to second tool gpt4_tool",
            func=get_vectorDBresponse
        )
        gpt4_tool = Tool.from_function(
            name="gpt4_tool",
            description="This is the Second tool to be used if you didn't find relevant answer from the first tool VectorDB_tool, This tool uses llm to generate a response, if you didn't find relevant answer then do not cook up data and without waisting time go to third tool serpapi_tool",
            func=get_gpt4_tool
        )
        serpapi_tool = Tool.from_function(
            name="serpapi_tool",
            description="This is the third tool to be used, use this tool if no tool is able to give the correct answer.This tool search the content from the web and brings complete response",
            func=get_serpresponse
        )     
        image_tool = Tool.from_function(
            name="image_tool",
            description="This tool is to be used for image related query and generation but only once in one go",
            func=generate_image_description
        )
       

        if st.button("Send") and query_input.strip():
            # Process user input
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.messages.append(HumanMessage(content=query, timestamp=timestamp, sender="user"))
            
            with st.spinner("Thinking..."):
                response_text = None
                tools= [vectorDB_tool,gpt4_tool,serpapi_tool,image_tool]
                agent = initialize_agent(
                    tools,
                    chat,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=6
                )
                # agent_executor = create_react_agent(chat,tools,checkpointer=memory)

                try:
                    response_text = agent.invoke(st.session_state.messages)
                    response_text = response_text['output']
                    print("response_text from agent=",response_text)
                    st.session_state.messages.append(response_text)
                except Exception as e:
                    response_text="I don't have enough information"
                    st.write(f"Error: {e}")


                # Append response and reset pending input
                st.session_state.messages.append(AIMessage(content=str(response_text), timestamp=timestamp, sender="AI"))
                st.session_state.pending_input = ""  # Reset pending input
                st.rerun()  # Refresh to show new messages

    # Chat history container (below input)
    chat_container = st.container()
    with chat_container:
        # Reverse order of messages for latest at the top
        messages = reversed(st.session_state.get('messages', []))  # Reverse the list
        for i, msg in enumerate(messages):  # Include all messages
            timestamp = msg.timestamp if hasattr(msg, "timestamp") else ""
            if isinstance(msg, AIMessage):
                message(f"{msg.content} ({timestamp})", is_user=False, key=f"{i}_ai")
            elif isinstance(msg, HumanMessage):
                message(f"{msg.content} ({timestamp})", is_user=True, key=f"{i}_user")

if __name__ == '__main__':
    main()
