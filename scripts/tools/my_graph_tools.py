from IPython.display import Image, display
import types

def view_graph(self):
    return Image(self.get_graph().draw_mermaid_png())

# Função para a chamada

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

import uuid
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

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

    output = self.invoke(initial_state, config=config)
    display_agent_output(output)

def turbinar_app(app):
    app.view = types.MethodType(view_graph, app)
    app.nice_call = types.MethodType(call, app)
    return app