def format_prompt(prompt: str, retrieved_documents: dict) -> str:
    PROMPT = (
        "You are GymRat, a helpful assistant specialized in fitness, "
        "workout routines, muscle growth, fat loss, and nutrition.\n\n"
        "Below are previously answered questions and their expert responses. "
        "Use them as context to answer the new question accurately.\n\n"
    )
    documents = retrieved_documents["metadatas"][0]
    for idx, doc in enumerate(documents):
        PROMPT += f"Example {idx + 1}:\n{doc['response']}\n\n"
    PROMPT += f"User Question: {prompt}\nAnswer:"
    return PROMPT
