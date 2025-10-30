import chainlit as cl
from app import app  # Make sure your LangGraph is compiled in app.py as `app`
from langchain_core.documents import Document  # Optional: for typing

@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Merhaba!").send()

@cl.on_message
async def on_message(message: cl.Message):
    user_question = message.content

    # Initial graph state
    inputs = {
        "question": user_question,
        "documents": [],
        "generation": "",
        "retry_count": 0
    }

    final_state = None

    # Stream through graph steps
    async for output in app.astream(inputs):
        for node_name, state in output.items():
            await cl.Message(content=f"🔹 Node çalıştırıldı: `{node_name}`").send()
            final_state = state  # Store the latest state


    # ✅ Show final result
    await cl.Message(
        content=f"{final_state.get('generation', '[Cevap üretilemedi]')}"
    ).send()
