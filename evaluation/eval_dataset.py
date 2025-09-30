"""
Defines the evaluation dataset for the RAG pipeline.

This dataset is based on the paul_graham_essay.txt file and is used by RAGAs
to quantitatively score the performance of the RAG system.
"""

from datasets import Dataset

def get_eval_dataset() -> Dataset:
    """
    Returns the evaluation dataset (questions and ground truths) for the RAG pipeline.
    """
    questions = [
        "What were the two main things the author worked on before college?",
        "What kind of stories did the author write, and what was his opinion of them?",
        "What did the author learn from Russian writers?",
        "How did the author's experience with programming influence his writing?",
        "What are the main lessons the author learned from writing and programming?"
    ]
    
    ground_truths = [
        ["Before college, the author mainly worked on writing and programming."],
        ["He wrote awful short stories that had hardly any plot, just characters with strong feelings."],
        ["From the Russian writers, he learned that it was possible to be serious in writing."],
        ["Programming taught him that, like fixing bugs in code, he could fix clunky sentences in his writing and improve by working at it."],
        ["The main lessons he learned were: to get good at something, you must do it a lot; it is important to be willing to fail; and you must be willing to work hard."]
    ]

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths
    })