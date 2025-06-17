import os
import zipfile
import operator
import types
import uuid
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
    try:
        tool_output = selected_tool.invoke(tool_call["args"])
    except:
        # try again
        try:
            tool_output = selected_tool.invoke(tool_call["args"])
        except Exception as e:
            tool_output = {"erro": e}
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

# Compilar app

app = stategraph.compile(checkpointer=checkpointer)

#Para poder passar o agente já com aschamadas personalizadas

def view_graph(self):
    return Image(self.get_graph().draw_mermaid_png())

def call(self, question: str):
    thread_id = uuid.uuid4() # identificação para memoria
    system = """
    You are a helpful assistant. Your job is to answer questions related to the contents of one or more CSV files
    that you have to extract from a .zip file. So before answering any question you need to find the zip file
    available to you, extarct it and only then you can try to answer the user inquiry.
    """
    initial_state = {"messages": [
            SystemMessage(system),
            HumanMessage(question)
        ]}
    config = {"configurable": {"thread_id": thread_id}}

    return self.invoke(initial_state, config=config)
       

def display_agent_output(agent_output):
    """
    Displays the output of a LangGraph agent in a clear and organized way.

    Args:
        agent_output (dict): The dictionary output from the LangGraph agent's invoke method.
    """
    if not isinstance(agent_output, dict) or 'messages' not in agent_output:
        print("Invalid agent output format. Expected a dictionary with a 'messages' key.")
        return

    print("--- AGENT EXECUTION LOG ---")

    for i, message in enumerate(agent_output['messages']):
        print(f"\n--- Message {i+1} ---")
        if isinstance(message, SystemMessage):
            print(f"  🤖 **System Message:**")
            print(f"    Content: {message.content.strip()}")
        elif isinstance(message, HumanMessage):
            print(f"  👤 **Human Message:**")
            print(f"    Content: {message.content.strip()}")
        elif isinstance(message, AIMessage):
            print(f"  🧠 **AI Message (Thought/Action):**")
            if message.content:
                print(f"    Content: {message.content.strip()}")
            if message.tool_calls:
                print("    Tool Calls:")
                for tool_call in message.tool_calls:
                    print(f"      - Name: {tool_call['name']}")
                    print(f"        Arguments: {tool_call['args']}")
                    print(f"        Tool Call ID: {tool_call['id']}")
            if message.response_metadata.get('finish_reason'):
                print(f"    Finish Reason: {message.response_metadata['finish_reason']}")
        elif isinstance(message, ToolMessage):
            print(f"  🛠️ **Tool Message (Tool Output):**")
            print(f"    Tool Call ID: {message.tool_call_id}")
            print(f"    Content: {message.content.strip()}")
        else:
            print(f"  ❓ **Unknown Message Type:** {type(message).__name__}")
            print(f"    Content: {message.content}")

    print("\n--- Summary ---")

    if 'intermediate_steps' in agent_output and agent_output['intermediate_steps']:
        print(f"👣 **Intermediate Steps (Node Execution Order):**")
        print(f"  {' -> '.join(agent_output['intermediate_steps'])}")

    if 'zip_files' in agent_output and agent_output['zip_files']:
        print(f"📦 **Identified Zip Files:** {', '.join(agent_output['zip_files'])}")

    if 'csv_files' in agent_output and agent_output['csv_files']:
        print(f"📄 **Extracted CSV Files:**")
        for csv_file in agent_output['csv_files']:
            print(f"  - {csv_file}")

    if 'final_answer' in agent_output and agent_output['final_answer']:
        print(f"✅ **Final Answer:** {agent_output['final_answer']}")
    else:
        print("🤷 No explicit final answer found in the output.")

    print("\n--- END OF AGENT LOG ---")

def nicecall(self, question: str):
    display_agent_output(call(self, question))

def turbinar_app(app):
    app.view = types.MethodType(view_graph, app)
    app.call = types.MethodType(call, app)
    app.nice_call = types.MethodType(nicecall, app)
    return app

def compilar_agente():
    turbo_app = turbinar_app(app)
    return turbo_app