#### single agent chatbot app using LangChain and Streamlit

import streamlit as st

# import libraries
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import time
from langchain_openai import ChatOpenAI
#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from tqdm import tqdm

load_dotenv()  # Load .env

# nnitialize LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "financial-consumer-chatbot"



# Initialize chat history
msgs = StreamlitChatMessageHistory()
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi! Ask me about consumer complaints, companies, or financial products.")

# Initialize retriever and LLM
embedding = OpenAIEmbeddings()
persist_directory = "chroma_db"

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
llm = ChatOpenAI(model="gpt-4o-mini")
retriever = vectorstore.as_retriever()

# Prompt to convert follow-up to standalone question
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# retreiver becomes history aware
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### wrap with message history

# store = {}
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# chat_history = StreamlitChatMessageHistory()
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# UI setup
st.set_page_config(page_title="Consumer Complaint Chatbot", layout="wide")
st.title("Consumer Complaint Explorer")
st.caption("Ask questions about real consumer finance complaints from the CFPB dataset.")

# loop through messages in StreamlitChatMessageHistory) and display them
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Display the chat input box for user input
query = st.chat_input("Ask a question about consumer complaints...")
if query:
    st.chat_message("human").write(query)
    with st.chat_message("ai"): # AI response
        start_time = time.time()
        with st.spinner("Searching and answering..."):
            result = conversational_rag_chain.invoke({"input": query}, config={"configurable": {"session_id": "default"}})
            response = result["answer"]
            sources = result.get("context", [])  # or use result["context"] if documents are exposed here
            st.markdown(response)

            # Debug: vector log
            elapsed = round(time.time() - start_time, 2)
            st.caption(f" Retrieved in {elapsed}s")

            # if sources:
            #     st.markdown("#### Top Sources")
            #     for i, doc in enumerate(sources[:3]):
            #         company = doc.metadata.get("Company", "Unknown")
            #         product = doc.metadata.get("Product", "Unknown")
            #         snippet = doc.page_content[:300].replace("\n", " ").strip()
            #         st.markdown(f"**{i+1}. {company} – {product}**")
            #         st.markdown(f"...{snippet}...")