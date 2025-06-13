import os
import re
import csv
import warnings
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

# Import the Pokemon API function
from pokemon_api import get_pokemon_weight

warnings.filterwarnings("ignore")

# API Key specific in .env file and environment variables loaded
load_dotenv()

# Valid humans for validation
valid_humans = {"ash", "misty", "brock", "jessie", "james"}

def setup_llm_and_embeddings():
    """Initialize LLM and embeddings to use"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable")
    
    # initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0  # for deterministic outputs (no randomness or creativity)
    )
    
    # initialize embeddings to use to create vector database
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    print("LLM and embeddings initialized")
    return llm, embeddings

def load_documents():
    """Load and split the Pokemon transcript document into chunks for easier processing"""
    
    # Load the transcript
    document = []
    doc = TextLoader("pokemon_transcript.txt").load()
    document.extend(doc)
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ": ", " ", ""]
    )
    
    chunks = splitter.split_documents(document)
    print(f"Loaded and split into {len(chunks)} chunks")

    return chunks

def create_vector_store(chunks, embeddings):
    """Create vector store with embeddings"""
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="pokemon_collection"
    )
    
    retriever = vectordb.as_retriever(
        search_type="similarity", # similarity search used for simplicity for retrieval
        search_kwargs={"k": 5}
    )
    print("Vector store created successfully")

    return retriever

def setup_memory():
    """Setup conversation memory"""
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return memory

def create_tools(retriever, llm):
    """Create tools with access to retriever and LLM"""
    
    @tool
    def search_pokemon_ownership(human_name):
        """Search for Pokemon owned by a specific human in the transcript using RAG retrieval"""
        try:
            # Use RAG retrieval to find relevant context
            query = f"What Pokemon does {human_name} own or have in their team?"
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Combine retrieved content
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Use LLM to extract Pokemon names instead of regex (regex did not work)
            extraction_prompt = f"""
            From the following text, extract the names of Pokemon owned by {human_name}.
            A Pokemon belongs to {human_name} if it is mentioned as being theirs, in their possession, on their team, or referred to with pronouns like "my" or "mine".
            Return only the Pokemon names as a comma-separated list.
            If no Pokemon are found, return "None".
            
            Text:
            {context}
            
            Pokemon owned by {human_name}:
            """
            
            response = llm.invoke(extraction_prompt)
            pokemon_names = response.content.strip()
            
            if pokemon_names.lower() == "none" or not pokemon_names:
                return f"No Pokemon found for {human_name}"
            else:
                return f"Found Pokemon for {human_name}: {pokemon_names}"
                
        except Exception as e:
            return f"Error searching for {human_name}'s Pokemon: {str(e)}"

    @tool
    def get_pokemon_weights(pokemon_names: str) -> str:
        """Get weights for multiple Pokemon using the Pokemon API"""
        try:
            # Clean up the input - remove "Found Pokemon for X:" prefix if present
            if ":" in pokemon_names:
                pokemon_names = pokemon_names.split(":", 1)[1].strip()
            
            names_list = [name.strip() for name in pokemon_names.split(',')]
            weights_data = []
            
            for pokemon_name in names_list:
                if pokemon_name:
                    weight_info = get_pokemon_weight(pokemon_name)
                    if isinstance(weight_info, dict) and 'text_description' in weight_info:
                        weights_data.append(f"{pokemon_name}: {weight_info['text_description']}")
                    else:
                        weights_data.append(f"{pokemon_name}: {weight_info}")
            
            return "\n".join(weights_data)
            
        except Exception as e:
            return f"Error getting Pokemon weights: {str(e)}"

    @tool
    def create_comparison_csv(human1_and_human2: str) -> str:
        """Create CSV file comparing Pokemon weights between two humans
        Expected input: 'human1 vs human2' or 'human1 and human2'
        """
        try:
            # Extract human names from input
            found_humans = []
            for human in valid_humans:
                if human.lower() in human1_and_human2.lower():
                    found_humans.append(human.capitalize())
            
            if len(found_humans) != 2:
                return "Error: Could not identify exactly 2 humans in input"
            
            human1, human2 = found_humans[0], found_humans[1]
            
            # Get Pokemon for each human separately using the search function
            pokemon1_result = search_pokemon_ownership.func(human1)
            pokemon2_result = search_pokemon_ownership.func(human2)
            
            if "No Pokemon found" in pokemon1_result or "No Pokemon found" in pokemon2_result:
                return f"Error: Could not find Pokemon for one or both humans\n{pokemon1_result}\n{pokemon2_result}"
            
            # Extract Pokemon names from search results
            pokemon1_names = pokemon1_result.split(":", 1)[1].strip() if ":" in pokemon1_result else pokemon1_result
            pokemon2_names = pokemon2_result.split(":", 1)[1].strip() if ":" in pokemon2_result else pokemon2_result
            
            # Get weights for each human's Pokemon
            weights1_result = get_pokemon_weights.func(pokemon1_names)
            weights2_result = get_pokemon_weights.func(pokemon2_names)
            
            # Parse weights from results
            weights1 = {}
            weights2 = {}
            
            def parse_weights(weight_result, weights_dict):
                for line in weight_result.split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        pokemon = parts[0].strip()
                        weight_text = parts[1].strip()
                        
                        # Extract weight value (handle both kg and hectograms)
                        import re
                        weight_match = re.search(r'([0-9.]+)', weight_text)
                        if weight_match:
                            weight_val = float(weight_match.group(1))
                            # Convert kg to hectograms if needed
                            if 'kg' in weight_text.lower():
                                weight_val *= 10
                            weights_dict[pokemon] = weight_val
            
            parse_weights(weights1_result, weights1)
            parse_weights(weights2_result, weights2)
            
            # Balance teams and create comparison
            min_count = min(len(weights1), len(weights2))
            if min_count == 0:
                return "Error: No valid Pokemon weights found"
            
            # Sort by weight and take top N
            sorted1 = sorted(weights1.items(), key=lambda x: x[1], reverse=True)[:min_count]
            sorted2 = sorted(weights2.items(), key=lambda x: x[1], reverse=True)[:min_count]
            
            # Create CSV and count wins
            csv_data = []
            wins1, wins2 = 0, 0
            
            for i in range(min_count):
                pokemon1, weight1 = sorted1[i]
                pokemon2, weight2 = sorted2[i]
                
                if weight1 > weight2:
                    winner = human1
                    wins1 += 1
                elif weight2 > weight1:
                    winner = human2
                    wins2 += 1
                else:
                    winner = "Tie"
                
                csv_data.append([
                    f"{pokemon1}({weight1})",
                    f"{pokemon2}({weight2})",
                    winner
                ])
            
            # Write CSV file
            filename = f"pokemon_battle_{human1}_vs_{human2}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([human1, human2, "Winner"])
                writer.writerows(csv_data)
            
            # Determine overall winner
            if wins1 > wins2:
                message = f"The winner of the duel between {human1} and {human2} is {human1}! Congratulations {human1}!! {human1} won against {human2} with a score of {wins1} to {wins2}!"
            elif wins2 > wins1:
                message = f"The winner of the duel between {human1} and {human2} is {human2}! Congratulations {human2}!! {human2} won against {human1} with a score of {wins2} to {wins1}!"
            else:
                message = "The competition is a tie!"
            
            return f"CSV file created: {filename}\n{message}"
            
        except Exception as e:
            return f"Error creating CSV: {str(e)}"
    
    return [search_pokemon_ownership, get_pokemon_weights, create_comparison_csv]

def setup_agent(llm, tools, memory):
    """Setup the ReAct agent with tools and memory"""
    
    # Create prompt template
    template = """
    You are a Pokemon Battle Analysis Agent. Your task is to determine the winner between two humans based on their Pokemon weights.

    Follow these steps:
    1. Validate that exactly 2 humans are mentioned and they are from: Ash, Misty, Brock, Jessie, James
    2. Use create_comparison_csv tool with the two human names to automatically:
       - Search for Pokemon owned by each human
       - Get their weights
       - Create comparison CSV and determine winner

    IMPORTANT: For create_comparison_csv, simply provide the two human names like:
    "Ash vs Misty" or "Brock and Jessie"

    Available tools: {tool_names}

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Current conversation:
    {chat_history}

    Question: {input}
    {agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["input", "chat_history", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        },
        template=template
    )
    
    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    print("Agent setup complete")

    return agent_executor

def validate_question(question: str) -> tuple[bool, str]:
    """Validate that the question contains exactly 2 valid humans"""
    question_lower = question.lower()
    found_humans = []
    
    for human in valid_humans:
        if human in question_lower:
            found_humans.append(human.capitalize())
    
    if len(found_humans) != 2:
        return False, "Incorrect question"
    
    return True, f"Valid question with humans: {found_humans[0]} and {found_humans[1]}"

def analyze_battle(agent_executor, question: str) -> str:
    """Main function to analyze Pokemon battle between two humans"""
    try:
        # Validate question
        is_valid, validation_msg = validate_question(question)
        if not is_valid:
            return validation_msg
        
        print(f"Processing question: {question}")
        print(f"Validation: {validation_msg}")
        
        # Run the agent
        result = agent_executor.invoke({"input": question})
        return result["output"]
        
    except Exception as e:
        return f"Error analyzing battle: {str(e)}"

def initialize_system():
    """Initialize the complete RAG system"""
    print("=== Pokemon RAG Battle Analysis Agent ===\n")
    print("Initializing system components...")
    
    llm, embeddings = setup_llm_and_embeddings()
    chunks = load_documents()
    retriever = create_vector_store(chunks, embeddings)
    memory = setup_memory()
    tools = create_tools(retriever, llm)
    agent_executor = setup_agent(llm, tools, memory)
    
    print("System initialization complete!\n")

    return agent_executor

def main():
    """Main execution function"""
    try:
        # Initialize the system
        agent_executor = initialize_system()
        
        # Test questions
        test_questions = [
            "Who would win between Ash and Misty?",
            "Compare Brock and Jessie in a Pokemon battle",
            "Who has stronger Pokemon: James or Ash?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"TEST {i}: {question}")
            print('='*50)
            
            result = analyze_battle(agent_executor, question)
            print(f"\nRESULT:\n{result}")
            
            print("\n" + "="*50)
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 