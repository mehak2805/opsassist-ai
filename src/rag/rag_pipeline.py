from src.rag.retriever import retrieve_documents


def get_context(query, top_k=2):
    results = retrieve_documents(query, top_k=top_k)

    if not results:
        return "", []

    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Relevance Score: {result['score']:.3f}\n"
            f"{result['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context, results