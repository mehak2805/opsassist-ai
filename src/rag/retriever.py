from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.knowledge_base.loader import load_documents


def retrieve_documents(query, top_k=2):
    documents = load_documents()

    if not documents:
        return []

    texts = [document["content"] for document in documents]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append(
            {
                "source": documents[index]["source"],
                "content": documents[index]["content"],
                "score": similarities[index],
            }
        )

    return results