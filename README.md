<div align="center">
  
  <h1>🌀 Multi-Source RAG Engine 🌀</h1>
  
  <p>
    <strong>Your intelligent assistant for documents, videos, and web pages. Ask questions in natural language and get precise, context-aware answers with cited sources.</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python Version">
    <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Framework">
    <img src="https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google" alt="LLM">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>

</div>

---

> Tired of sifting through hours of video lectures, dense research papers, or lengthy articles to find one specific piece of information? This tool solves that. Simply provide a source, and start a conversation.

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that transforms static content into a dynamic conversational partner. It leverages the power of Google's Gemini models and LangChain to provide accurate, fast, and transparent insights.

<div align="center">
  <img src="placeholder.gif" alt="App Demo GIF" width="800"/>
</div>

##   Core Features

-   **Multi-Source Intelligence**: Ingest and process content from various sources:
    -   **YouTube**: Provide a video URL to chat with its transcript.
    -   **Web Pages**: Enter any URL to analyze its content.
    -   **Documents(supports multiple PDFs/Docxs files)**: Upload your own PDF (`.pdf`) and Word (`.docx`) files.
-   **Interactive & Fluid Chat**:
    -   **Streaming Responses**: Get answers word-by-word in real-time for a dynamic feel.
    -   **Conversation History**: Maintains context throughout the conversation.
-   **AI-Powered Assistance**:
    -   **Suggested Questions**: The AI automatically generates insightful questions to kickstart the conversation.
-   **(Transparent) & Trustworthy**:
    -   **Source Citations**: Every answer is backed by the source text it was derived from. Just expand the "Show Sources" section.
-   **Blazing Fast & Efficient**:
    -   **Intelligent Caching**: All expensive operations are cached, making subsequent queries instantaneous.
    -   **Robust State Management**: Chat sessions are isolated, preventing data leaks between different sources.

##   How It Works: The RAG Pipeline

The application follows a sophisticated pipeline to deliver accurate answers:

1.  **Data Ingestion**: The user provides a source (URL or file) via the Streamlit sidebar.
2.  **Data Loading**: LangChain loaders (`PyPDF`, `BeautifulSoup`, `YoutubeTranscriptApi`, etc.) extract raw text.
3.  **Text Chunking**: The extracted text is split into smaller, semantically meaningful chunks.
4.  **Embedding Generation**: Each chunk is converted into a numerical vector representation (embedding) using `Sentence-Transformers`.
5.  **Vector Storage**: The embeddings are indexed and stored in a `ChromaDB` vector store for efficient retrieval.
6.  **User Query**: The user asks a question.
7.  **Similarity Search**: The user's query is embedded, and a similarity search is performed in ChromaDB to find the most relevant text chunks (the "context").
8.  **Context Augmentation**: The relevant context and the user's query are combined into a detailed prompt for the LLM.
9.  **LLM Response**: The prompt is sent to the **Google Gemini** model, which generates a natural language answer based *only* on the provided context.
10. **Stream to UI**: The generated response is streamed back to the Streamlit interface for the user to see.

##   Tech Stack

| Category          | Technology                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **Frontend** | Streamlit                                                                                                   |
| **Backend** | Python                                                                                                      |
| **AI Frameworks** | LangChain, LangChain Community                                                                              |
| **LLM** | Google Gemini (via `langchain-google-genai`)                                                                |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`)                                                                  |
| **Vector DB** | ChromaDB                                                                                                    |
| **Data Loaders** | `pypdf`, `python-docx`, `beautifulsoup4`, `youtube-transcript-api`                                            |
| **Transcription** | `faster-whisper`, `yt-dlp`, `pydub`                                                                         |

##   Getting Started

Get the application running on your local machine in just a few steps.

#### 1. Prerequisites

-   Python 3.9+
-   A **Google Gemini API Key**. You can get one for free from [Google AI Studio](https://makersuite.google.com/app/apikey).

#### 2. Clone the Repository

```bash
git clone https://github.com/The-Name-is-Karthik/Multi-Source-RAG-Engine.git
cd Multi-Source-RAG-Engine
```

#### 3. Install Dependencies
Create a virtual environment and install all the required packages.
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

#### 4. Set Up Environment Variables
Create a file named .env in the root directory of the project and add your API key.
```bash
GEMINI_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

#### 5. Run the Application
Launch the Streamlit app.
```bash
streamlit run app.py
```


#### Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1\.  Fork the Project

2\.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)

3\.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)

4\.  Push to the Branch (`git push origin feature/AmazingFeature`)

5\.  Open a Pull Request


<div align="center">
Made with ❤️ by Karthik
</div>












<!--



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

-   [ ] Persist the vector store to disk to remember documents between sessions. -->
