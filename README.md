# Multi-Source-RAG-Engine

Multi-Source-RAG-Engine is an intelligent, multi-source AI application that allows you to chat with your content—whether it's a YouTube video, a web page, or a dense document(PDF/DOCX).

## The Problem

In a world of information overload, we often need specific answers from long-form content without spending hours watching, reading, or scrolling. Manually finding these key insights is tedious and inefficient.

## The Solution

Multi-Source-RAG-Engine solves this by acting as a powerful knowledge assistant. It ingests content from multiple sources, creates a searchable knowledge base, and uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers. This allows users to have a direct conversation with their documents and videos, extracting valuable information in seconds.

## Key Features
---------------

-   **Multi-Source Ingestion**: Seamlessly processes content from YouTube videos, public web pages, PDFs, and DOCX files.

-   **Robust Transcription**: Features a graceful fallback from YouTube's transcript API to local audio transcription via the Whisper model.

-   **Efficient & Stateful UI**: Caches processed data to prevent costly re-embedding on every query and maintains separate, independent chat histories for each content source.

-   **Grounded & Factual Answers**: Leverages a RAG pipeline to ensure the AI's responses are based strictly on the provided source material, preventing hallucinations.

---

## Technical Architecture
-------------------------

The application is built on a modular, three-stage pipeline that ensures separation of concerns and scalability.

1.  **Data Ingestion Layer**: A universal data loader uses specialized parsers (`youtube-transcript-api`, `WebBaseLoader`, `PyPDFLoader`) to extract raw text from any source.

2.  **Indexing Layer**: The text is chunked, converted into vector embeddings using `Sentence-Transformers`, and stored in an in-memory `ChromaDB` vector store.

3.  **Retrieval & Generation Layer**: When a user asks a question, the most relevant chunks are retrieved from the database. This context, along with the user's query, is passed to an `OpenAI` LLM via a carefully engineered prompt to generate the final answer.


---

## Tech Stack
-------------

-   **Language**: Python

-   **AI Framework**: LangChain

-   **LLM**: Gemini-2.5-flash

-   **UI**: Streamlit

-   **Vector Database**: ChromaDB (in-memory)

-   **Embeddings**: Sentence-Transformers (local model: `all-MiniLM-L6-v2`)

-   **Data Processing**: `yt-dlp`, `pypdf`, `python-docx`, `BeautifulSoup4`

---

## How to Run Locally
---------------------

Follow these steps to set up and run the project on your local machine.

**1\. Clone the repository:**

```bash
git clone https://github.com/The-Name-is-Karthik/Muti-Source-RAG-Engine.git
cd Muti-Source-RAG-Engine
```
**2\. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3\. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4\. Setup environment variables:**
Create a .`env` file in the root directory
Add your Gemini API key: `GEMINI_API_KEY=`


**5\. Run the application:**
```bash
streamlit run app.py
```


---
## Future Improvements
----------------------

-   [ ] Implement asynchronous streaming for real-time LLM responses.

-   [ ] Add support for more document types (e.g., `.csv`, `.pptx`).

-   [ ] Persist the vector store to disk to remember documents between sessions.