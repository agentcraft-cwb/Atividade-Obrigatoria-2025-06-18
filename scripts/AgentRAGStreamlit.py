import streamlit as st
import asyncio
import uuid
from langchain_core.messages import HumanMessage

st.title("GraphRAG Agent")

from GraphRAGAgent import compilar_agente
app = compilar_agente()

user_map = {
    "human": "user",
    "ai": "assistant"
}
avatar_map = {
    "user": "images/user.png",
    "assistant": "images/agentcraft_icon.png"
}

#thread_id = None
#config = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "config" not in st.session_state:
    st.session_state.config = None

async def invoke(prompt, status_placeholder_container):
    
    if st.session_state.config:
        initial_state = {"messages": HumanMessage(prompt)}
    else:
        st.session_state.thread_id = uuid.uuid4()
        st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial_state = {
            "ignore_msgs": 0,
            "token_count": 0,
            "cost_count": 0,
            "messages": HumanMessage(prompt)}
        
    with status_placeholder_container.status("Silênco... agente pensando!", expanded=False) as status_container:
        async for event in app.astream_events(initial_state, config=st.session_state.config, version="v2", stream_mode="custom"):
            if "chunk" in event['data']: 
                if event["data"]["chunk"] is not None:
                    if "custom_key" in event["data"]["chunk"]:
                        status_container.update(label = event["data"]["chunk"]["custom_key"]['status'])
                        status_container.write(event["data"]["chunk"]["custom_key"]['node'] + "->" + event["data"]["chunk"]["custom_key"]['status'])
    messages = []
    for message in event['data']['output']['messages']:
        messages.append({"role": user_map[message.type], "content": message.content})

    return messages

messages_container = st.container(border=True, height=350)

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

    status_placeholder = st.empty() 

    response = asyncio.run(invoke(prompt, status_placeholder))

    content = response[-1]['content']
    # Display assistant response in chat message container
    with messages_container.chat_message("assistant", avatar=avatar_map["assistant"]):
        messages_container.markdown(content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": content})

