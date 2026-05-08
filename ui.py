import chainlit as cl
# We import your existing logic from app.py
from app import build_rag_prompt, stream_answer_chunks

@cl.on_chat_start
async def start():
    # This runs when you first open the webpage
    cl.user_session.set("top_k", 3)
    await cl.Message(content="🚀 Project RAG Demo is online. How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    system_prompt, user_prompt, sources = await cl.make_async(build_rag_prompt)(
        message.content,
        cl.user_session.get("top_k"),
    )

    for token in stream_answer_chunks(system_prompt, user_prompt):
        await msg.stream_token(token)

    msg.elements = [
        cl.Text(name=f"Source {i+1}", content=s, display="side") 
        for i, s in enumerate(sources)
    ]
    await msg.update()
