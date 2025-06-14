import os
import zipfile
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from IPython.display import Image, display
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = "gpt-4o"

# Criação e declaração das ferramentas do agente
# - listar arquivos .zip no diretório de trabalho
# - extrair arquivos .zip para a pasta '.\datasets'
# - sub agente para fazer CSV RAG nos arquivos

llm = ChatOpenAI(model=MODEL_NAME)

@tool
def answer_question(question: str, csv_files: list[str]):
    """
    Answer the question posed by the user.

    args:
    - question: The question or inquiry posed by the used
    - csv_files: the .csv files that it consults to find the answer
    """
    agent_csv = create_csv_agent(llm, csv_files, verbose=False, allow_dangerous_code=True)
    output = agent_csv.invoke(question)
    return {
        "final_answer": output['output']
        }

@tool
def extract_zip_files(local_zip_filepaths:list[str]) -> list:
    """
    Extracts files listed in 'local_zip_filepaths'                                    
    """
    extract_path = "./datasets"
    os.makedirs(extract_path, exist_ok=True)

    try:
        for local_zip_filepath in local_zip_filepaths:
            with zipfile.ZipFile(local_zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        contents = [os.path.join(extract_path, file) for file in os.listdir(extract_path)]
        return { "csv_files": contents}
    except zipfile.BadZipFile:
        return { "error": f"The downloaded file is not a valid ZIP file."}
    except Exception as e:
        return { "error": f"An error occurred during extraction: {e}"}

@tool    
def list_available_zip_files() -> list:
    """
    Lists all zip files available for extraction.
    """
    zip_files = []
    for filename in os.listdir('.'):  # '.' 
        if filename.endswith('.zip') and os.path.isfile(filename):
            zip_files.append(filename)
    return {"zip_files": zip_files}

tools = [extract_zip_files, list_available_zip_files, answer_question]

#Criação llm com 'binded tools'
llm_tools = llm.bind_tools(tools)

# Definição da estado/memóra

checkpointer = MemorySaver()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intermediate_steps: Annotated[list[str], operator.add]
    zip_files: list[str]
    csv_files: list[str]
    final_answer: str

#Definição dos nodes

def main_assistant(state:AgentState)->AgentState:
    """
    Assistente orquestrador. Recebe o estado inicial com System e User prompt e
    orquestra a execução das chamadas de ferramenta
    """  
    messages = state.get('messages')
    agent_outcome = llm_tools.invoke(messages)
    return {
        "messages": [agent_outcome],
        "intermediate_steps": ["main_agent"]
        }

def execute_tool(state:AgentState)->AgentState:
    """
    Executa as ferramentas chamadas. Node que cada ferramenta já retorna no formato 
    correto para atualizar o 'state'.
    """
    tools_map = {
            "extract_zip_files": extract_zip_files, 
            "list_available_zip_files": list_available_zip_files,
            "answer_question": answer_question
        }
    messages = state.get('messages')
    ai_msg = messages[-1]

    tool_output = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = tools_map[tool_call["name"]]

    tool_output = selected_tool.invoke(tool_call["args"])
    tool_output['intermediate_steps'] = [tool_call["name"]]
    tool_output['messages'] = [ToolMessage(tool_output, tool_call_id=tool_call['id'] )]

    return tool_output

state =AgentState()
stategraph = StateGraph(AgentState)

# Nodes
stategraph.add_node("main_assistant", main_assistant)
stategraph.add_node("tool_node", execute_tool)

# Edges

# Condicional entre orquestrador e executor
def should_continue(state: AgentState):
    """
    Guia o agente para o executor de ferramentas ou para o fim
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

stategraph.add_edge(START, "main_assistant")

stategraph.add_conditional_edges(
    "main_assistant",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    },
)

stategraph.add_edge("tool_node", "main_assistant")

# Call the function to display the output

def compliar_agente():
    return stategraph.compile(checkpointer=checkpointer)

