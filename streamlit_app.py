import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.title('👾 Pokemon Battle Analyzer')
st.info('This app builds a RAG agent for Pokemon Battle Analysis!')

with st.expander('Dataset'):
    st.write('**Raw Data**')
    
    try:
        # Load the transcript
        with open("pokemon_transcript.txt", "r") as f:
            transcript_content = f.read()
        
        st.text_area("Pokemon Transcript", transcript_content, height=200)
        
    except FileNotFoundError:
        st.error("pokemon_transcript.txt file not found!")
