from pathlib import Path


KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")


def load_documents():
    documents = []

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        print(f"- {document['source']}")