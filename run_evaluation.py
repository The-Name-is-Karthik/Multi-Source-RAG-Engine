"""
End-to-End RAG Evaluation Script using RAGAs.

This script performs the following steps:
1.  Loads a ground-truth document (Paul Graham's essay).
2.  Loads an evaluation dataset of questions and ideal answers.
3.  Initializes the RAG pipeline (data loading, vector store, and chain).
4.  Runs each question through the RAG pipeline to get the generated answer and retrieved context.
5.  Uses the RAGAs framework to evaluate the pipeline's performance based on the generated outputs
    and the ground-truth data.
6.  Calculates key metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
7.  Prints a comprehensive report of the evaluation results.

To run this script:
1. Make sure you have the necessary dependencies: `pip install -r requirements.txt`
2. Set your `GEMINI_API_KEY` in a `.env` file.
3. Run the script from the root of the project: `python run_evaluation.py`
"""

import os
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper as LangchainLLM

from langchain.schema.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


from src.vector_store import create_vector_store
from evaluation.eval_dataset import get_eval_dataset
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to run the RAG evaluation."""
    # --- 1. Load Data and Evaluation Set ---
    logging.info("Loading ground truth document and evaluation dataset...")
    with open("paul_graham_essay.txt", "r") as f:
        essay_text = f.read()
    
    docs = [Document(page_content=essay_text)]
    eval_dataset = get_eval_dataset()

    # --- 2. Build a Simple RAG Chain for Evaluation ---
    # Note: We are not using the conversational chain from the app to keep the evaluation focused.
    logging.info("Building a non-conversational RAG chain for evaluation...")
    
    vector_store = create_vector_store(docs)
    retriever = vector_store.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY)

    # The prompt for evaluation focuses on direct answering from context
    eval_prompt_template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Be concise.

    Question: {input}
    Context: {context}
    Answer:"""
    
    prompt = PromptTemplate.from_template(eval_prompt_template)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- 3. Run Pipeline and Collect Results ---
    logging.info("Running RAG pipeline on evaluation questions...")
    results = []
    for item in eval_dataset:
        question = item["question"]
        response = rag_chain.invoke({"input": question})
        results.append({
            "question": question,
            "answer": response["answer"],
            "contexts": [doc.page_content for doc in response["context"]],
        })

    # Convert results to a Hugging Face Dataset
    results_dataset = Dataset.from_list(results)
    
    # Merge the results with the original ground truths
    ground_truths = [item[0] for item in eval_dataset["ground_truth"]]
    eval_dataset_with_results = results_dataset.add_column("ground_truth", ground_truths)
    
    # --- 4. Evaluate with RAGAs ---
    logging.info("Evaluating the results using RAGAs...")
    
    metrics = [
        faithfulness,       # How factually consistent is the answer with the context?
        answer_relevancy,   # How relevant is the answer to the question?
        context_precision,  # Signal-to-noise ratio of the retrieved context.
        context_recall,     # Does the context contain all necessary info to answer the question?
    ]
    
    # Configure RAGAs to use Gemini and the same embeddings
    ragas_llm = LangchainLLM(llm)
    
    result = evaluate(
        dataset=eval_dataset_with_results,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=vector_store._embedding_function,  #Use the same embeddings as the retriever
    )

    # --- 5. Print Report ---
    logging.info("Evaluation complete. Generating report...")
    
    # Convert the evaluated dataset to a pandas DataFrame
    df = eval_dataset_with_results.to_pandas()
    res_df = result.to_pandas()

    df[res_df.columns] = res_df


    print("\n\n--- RAGAs Evaluation Report ---")
    print("\nThis report provides a quantitative measure of your RAG pipeline's performance.")
    print("Higher scores are better (max is 1.0).\n")
    
    print(f"Overall Scores:\n"
          f"  - Faithfulness:      {df['faithfulness'].mean():.4f}\n"
          f"  - Answer Relevancy:  {df['answer_relevancy'].mean():.4f}\n"
          f"  - Context Precision: {df['context_precision'].mean():.4f}\n"
          f"  - Context Recall:    {df['context_recall'].mean():.4f}")
    
    print("\n--- Per-Question Details ---")
    print(df[['question', 'answer', 'ground_truth', 'faithfulness', 'answer_relevancy']].to_string())
    print("\nFor a detailed breakdown and more metrics, inspect the 'result' object.")
    print("-----------------------------")

if __name__ == "__main__":
    # Ensure API key is available
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
    main()