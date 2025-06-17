import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

def display_agent_output(agent_output): #Formating the output to the status window
    """
    Displays the output of a LangGraph agent in a clear and organized way.

    Args:
        agent_output (dict): The dictionary output from the LangGraph agent's invoke method.
    """
    output = ""
    if not isinstance(agent_output, dict) or 'messages' not in agent_output:
        output = "Invalid agent output format. Expected a dictionary with a 'messages' key."
        return

    output +="--- AGENT EXECUTION LOG ---"

    for i, message in enumerate(agent_output['messages']):
        output += f"\n\n--- Message {i+1} ---"
        if isinstance(message, SystemMessage):
            output += f"\n  🤖 **System Message:**"
            output += f"\n    Content: {message.content.strip()}"
        elif isinstance(message, HumanMessage):
            output += f"\n  👤 **Human Message:**"
            output += f"\n    Content: {message.content.strip()}"
        elif isinstance(message, AIMessage):
            output += f"\n  🧠 **AI Message (Thought/Action):**"
            if message.content:
                output += f"\n    Content: {message.content.strip()}"
            if message.tool_calls:
                output += "\n    Tool Calls:"
                for tool_call in message.tool_calls:
                    output += f"\n      - Name: {tool_call['name']}"
                    output += f"\n        Arguments: {tool_call['args']}"
                    output += f"\n        Tool Call ID: {tool_call['id']}"
            if message.response_metadata.get('finish_reason'):
                output += f"\n    Finish Reason: {message.response_metadata['finish_reason']}"
        elif isinstance(message, ToolMessage):
            output += f"\n  🛠️ **Tool Message (Tool Output):**"
            output += f"\n    Tool Call ID: {message.tool_call_id}"
            output += f"\n    Content: {message.content.strip()}"
        else:
            output += f"\n  ❓ **Unknown Message Type:** {type(message).__name__}"
            output += f"\n    Content: {message.content}"

    output += "\n\n--- Summary ---"

    if 'intermediate_steps' in agent_output and agent_output['intermediate_steps']:
        output += f"\n👣 **Intermediate Steps (Node Execution Order):**"
        output += f"\n  {' -> '.join(agent_output['intermediate_steps'])}"

    if 'zip_files' in agent_output and agent_output['zip_files']:
        output += f"\n📦 **Identified Zip Files:** {', '.join(agent_output['zip_files'])}"

    if 'csv_files' in agent_output and agent_output['csv_files']:
        output += f"\n📄 **Extracted CSV Files:**"
        for csv_file in agent_output['csv_files']:
            output += f"\n  - {csv_file}"

    if 'final_answer' in agent_output and agent_output['final_answer']:
        output += f"\n✅ **Final Answer:** {agent_output['final_answer']}"
    else:
        output += f"\n🤷 No explicit final answer found in the output."

    output += f"\n\n--- END OF AGENT LOG ---"
    return output

st.title("LangGraph simple agent")
st.text("As respostas finais são mostradas na janela de chat. As etapas são detalhadas no drop-down de status, uma fez finalizado o processo.")
st.badge("chat box:", icon="💬", color="green")
from LangGraphAgent import compilar_agente
app = compilar_agente()

user_map = {
    "human": "user",
    "ai": "assistant"
}
avatar_map = {
    "user": "images/user.png",
    "assistant": "images/agentcraft_icon.png"
}

if "config" not in st.session_state:
    st.session_state.config = None

def invoke(prompt, status_container):
   
    with status_container as sc:
        sc.update(label = "Agente pensando 🤯", state="running")

        output = app.call(prompt)
        status_output = display_agent_output(output)
    
    with status_container as sc:
        sc.update(label = "Tarefa concluida 🤥🤣", state="complete")
        sc.write(status_output)
    message = {"role": "assistant", "content": output['messages'][-1].content}

    return message

messages_container = st.container(border=True, height=310)
st.badge("agent status:", icon="🥸", color="red")
status_container = st.status(label="Agente pronto e esperando 🥱", state="complete", expanded=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with messages_container.chat_message(message["role"], avatar=avatar_map[message["role"]]):
        messages_container.markdown(message["content"])

if prompt := st.chat_input("Prompt do usuário"):
    # Display user message in chat message container
    with messages_container.chat_message("user", avatar=avatar_map["user"]):
        messages_container.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = invoke(prompt, status_container)
    
    # Display assistant response in chat message container
    with messages_container.chat_message("assistant", avatar=avatar_map["assistant"]):
        messages_container.markdown(response['content'])
    # Add assistant response to chat history
    st.session_state.messages.append(response)

