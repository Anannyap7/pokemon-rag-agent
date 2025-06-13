import streamlit as st
import os
from dotenv import load_dotenv

# Fix protobuf compatibility issue
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
if st.button("Initialize RAG System"):
    with st.spinner("Initializing RAG system..."):
        try:
            # Import heavy libraries only when needed
            from langchain_community.document_loaders import TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            from langchain_chroma import Chroma
            
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
            
            # Create vector store
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db",
                collection_name="pokemon_collection"
            )
            
            st.success(f"RAG system initialized successfully! Created {len(chunks)} document chunks.")
            st.session_state.rag_initialized = True
            st.session_state.vectordb = vectordb
            
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
