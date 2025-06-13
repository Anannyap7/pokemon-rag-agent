import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

st.title('👾 Pokemon Battle Analyzer')
st.info('This app builds a RAG agent for Pokemon Battle Analysis!')

with st.expander('Dataset'):
    st.write('**Raw Data**')
    
    try:
        # Load the transcript
        document = []
        doc = TextLoader("pokemon_transcript.txt").load()
        document.extend(doc)
        
        # Display the document content
        if document:
            st.success(f"✅ Successfully loaded {len(document)} document(s)")
            
            # Display the content of the first document
            st.write("**Document Content:**")
            st.text_area("Pokemon Transcript", document[0].page_content, height=300)
            
            # Show document metadata
            st.write("**Document Metadata:**")
            st.json(document[0].metadata)
            
        else:
            st.warning("⚠️ No documents loaded")
            
    except FileNotFoundError:
        st.error("❌ pokemon_transcript.txt file not found. Please make sure the file exists in the current directory.")
    except Exception as e:
        st.error(f"❌ Error loading document: {str(e)}")
