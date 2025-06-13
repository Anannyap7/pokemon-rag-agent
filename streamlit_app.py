import streamlit as st
import os
from dotenv import load_dotenv

# Fix protobuf compatibility issue
# This environment variable tells the protobuf library to use the pure Python implementation instead of the C++ implementation
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

load_dotenv()

st.title('👾 Pokemon Battle Analyzer')
st.info('This app builds a RAG agent for Pokemon Battle Analysis!')

# Check if API key is available
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

# To initialize and display the dataset.
with st.expander('Dataset'):
    st.write('**Raw Data**')
    
    try:
        # Load the transcript
        with open("pokemon_transcript.txt", "r") as f:
            transcript_content = f.read()
        
        st.text_area("Pokemon Transcript", transcript_content, height=200)
        
    except FileNotFoundError:
        st.error("pokemon_transcript.txt file not found!")

# Only import and initialize heavy components when needed
# Creates a clickable button. The if statement only executes when the button is pressed.
if st.button("Initialize RAG System"):
    # Shows a spinning loading indicator with custom text
    with st.spinner("Initializing RAG system..."):
        try:
            # Import libraries
            from langchain_community.document_loaders import TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS
            
            # Initialize components
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=api_key,
                temperature=0
            )
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            # Load and process documents
            document = []
            doc = TextLoader("pokemon_transcript.txt").load()
            document.extend(doc)
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ": ", " ", ""]
            )
            
            chunks = splitter.split_documents(document)
            
            # Create vector store using FAISS instead of ChromaDB
            vectordb = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            # Save the vector store
            vectordb.save_local("faiss_index")
            
            st.success(f"RAG system initialized successfully! Created {len(chunks)} document chunks using FAISS.")
            # Streamlit's way to store data between app reruns. When a user interacts with the app (clicks button, types text), Streamlit reruns the entire script. Session state preserves variables across these reruns.
            st.session_state.rag_initialized = True # boolean flag object
            st.session_state.vectordb = vectordb # vector database object
            st.session_state.llm = llm # language model object
            
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")

# Battle analysis interface
if st.session_state.get('rag_initialized', False):
    st.subheader("Pokemon Battle Analysis")

    # Creates a single-line text input field
    question = st.text_input("Ask about Pokemon battles:", 
                           placeholder="Who would win between Ash and Misty?") # Gray hint text that appears inside the empty input field
    
    if st.button("Analyze Battle") and question:
        with st.spinner("Analyzing battle..."):
            try:
                # Import the main analysis function
                from pokemon_rag_agent import initialize_system, analyze_battle
                
                # Initialize the agent
                agent_executor = initialize_system() # Sets up the complete RAG pipeline (LLM, embeddings, tools, agent)
                
                # Analyze the battle
                result = analyze_battle(agent_executor, question)
                
                st.write("**Analysis Result:**")
                st.write(result)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}") 
