from langchain_core.tools import tool


def make_retriever(index):
    @tool
    def search_policies(question: str) -> list[dict]:
        """Retrieve relevant passages from the provided university IT policy file."""
        return [
            {"id": doc.metadata["policy_id"], "topic": doc.metadata["topic"],
             "source": doc.metadata["source"], "text": doc.page_content}
            for doc in index.retrieve(question)
        ]

    return search_policies
