import logging
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain.schema.document import Document
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data
def load_from_webpage(url: str) -> List[Document]:
    """Loads text from a web page."""
    logging.info(f"Loading content from URL: {url}")
    loader = WebBaseLoader(url)
    return loader.load()

@st.cache_data
def load_from_pdf(file_path: str) -> List[Document]:
    """Loads text from a PDF file."""
    logging.info(f"Loading content from PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_data
def load_from_docx(file_path: str) -> List[Document]:
    """Loads text from a DOCX file."""
    logging.info(f"Loading content from DOCX: {file_path}")
    loader = Docx2txtLoader(file_path)
    return loader.load()