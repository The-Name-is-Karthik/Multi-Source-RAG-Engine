# import logging
# import streamlit as st
# from langchain_community.vectorstores.chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_google_genai import ChatGoogleGenerativeAI

# from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @st.cache_resource
# def create_rag_chain(_retriever: Chroma.as_retriever):
#     """
#     Creates a conversational RAG chain with history and source retrieval.
#     """
#     try:
#         llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY)
#         logging.info(f"{GEMINI_MODEL_NAME} model initialized successfully.")
#     except Exception as e:
#         logging.error(f"Failed to initialize Gemini model: {e}")
#         raise

#     contextualize_q_system_prompt = """Given a chat history and the latest user question \
#     which might reference context in the chat history, formulate a standalone question \
#     which can be understood without the chat history. Do NOT answer the question, \
#     just reformulate it if needed and otherwise return it as is."""

#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
    
#     history_aware_retriever = create_history_aware_retriever(
#         llm, _retriever, contextualize_q_prompt
#     )

#     # qa_system_prompt = """You are an expert assistant. Answer the user's question based ONLY on the provided context. \
#     # If the information is not in the context, say "I don't know, the information is not available in the provided source." \
#     # Do not make up information. Be concise and helpful.

#     # CONTEXT:
#     # {context}
#     # """
    
#     qa_system_prompt = """You are an expert assistant. 
#     Answer using the provided context when possible. 

#     - If answer is in the context: start with "Based on the provided context: ..."
#     - If not in the context: use general knowledge, start with "Based on general knowledge: ..."
#     - If unsure: say "I don't know."
#     - Never invent facts. Be concise and clear.

#     CONTEXT:
#     {context}
#     """



#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", qa_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
#     logging.info("Conversational RAG chain created successfully.")
#     return rag_chain



import logging
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.chroma import Chroma

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def create_rag_chain(_retriever: Chroma.as_retriever):
    """
    Creates a conversational RAG chain with history and source retrieval.
    """
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY)
        logging.info(f"{GEMINI_MODEL_NAME} model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        raise

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, _retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an expert assistant. 
Answer using the provided context when possible. 

- If answer is in the context: start with "Based on the provided context: ..."
- If not in the context: use general knowledge, start with "Based on general knowledge: ..."
- If unsure: say "I don't know."
- Never invent facts. Be concise and clear.

CONTEXT:
{context}
"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    logging.info("Conversational RAG chain created successfully.")
    return rag_chain
