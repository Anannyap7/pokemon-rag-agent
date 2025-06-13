# 👾 Pokemon Battle Analyzer

A RAG (Retrieval-Augmented Generation) powered Pokemon battle analysis system that determines winners based on Pokemon weights using AI agents and vector search.\\
**Valid Names: Ash, Misty, James, Jessie, Brock.**

## 🚀 Features

- **RAG-Powered Analysis**: Uses FAISS vector database for semantic search through Pokemon transcripts
- **AI Agent System**: LangChain ReAct agents with custom tools for Pokemon data extraction
- **Weight-Based Battles**: Fetches real Pokemon weights via PokeAPI for fair comparisons
- **Interactive Web UI**: Beautiful Streamlit interface with real-time analysis
- **CSV Battle Reports**: Generates detailed comparison tables with downloadable results
- **Battle History**: View and compare previous battle results
- **Pronoun Recognition**: Smart detection of Pokemon ownership including "my" and "mine" references

## 🛠️ Tech Stack

- **LangChain**: Agent framework and document processing
- **Google Gemini**: LLM for Pokemon extraction and analysis
- **FAISS**: Vector database for semantic search
- **Streamlit**: Interactive web interface
- **PokeAPI**: Real Pokemon data source
- **Pandas**: Data manipulation and CSV handling

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini)
- Internet connection (for PokeAPI)

## ⚡ Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd pokemon-rag-agent
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```

3. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Use the System**
   - Click "Initialize RAG System" to set up the vector database
   - Ask battle questions like "Who would win between Ash and Misty?"
   - View detailed CSV battle reports and download results
