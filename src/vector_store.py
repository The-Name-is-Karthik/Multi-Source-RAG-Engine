import logging
from typing import List
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -%(filename)s - %(message)s')

@st.cache_resource
def create_vector_store(documents: List[Document]) -> Chroma:
    """
    Creates a Chroma vector store from a list of Document objects.
    """
    if not documents:
        logging.error("Input documents list for vector store creation is empty.")
        raise ValueError("Cannot create vector store from empty documents list.")

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        logging.error("Document splitting resulted in no chunks.")
        raise ValueError("Documents could not be split into chunks.")
    logging.info(f"Documents split into {len(chunks)} chunks.")

    logging.info("Initializing embedding model...")
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    logging.info("Creating vector store...")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    logging.info("Vector store created successfully.")

    return vector_store