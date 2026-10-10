import argparse
import hashlib
import json
from pathlib import Path

from policy_loader import load_policy_documents
from thermal_guard import wait_for_gpu

HOME = Path(__file__).resolve().parent
EMBEDDER = "nomic-embed-text"


class PolicyIndex:
    def __init__(self, folder=None, endpoint="http://127.0.0.1:11434", database=None):
        self.documents = load_policy_documents()
        self.sources = {doc.metadata["policy_id"]: doc for doc in self.documents}
        if database is not None:
            self.database = database
            return
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        folder = Path(folder or HOME / ".index")
        folder.mkdir(parents=True, exist_ok=True)
        entries = [(doc.page_content, doc.metadata) for doc in self.documents]
        signature = {
            "model": EMBEDDER,
            "source_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        }
        manifest = folder / "source.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != signature:
                raise ValueError("This index belongs to a different source or embedding model")
        elif any(folder.iterdir()):
            raise ValueError("An existing index without its source record cannot be reused")
        else:
            manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        embeddings = OllamaEmbeddings(
            model=EMBEDDER, base_url=endpoint, num_gpu=99, num_thread=2, num_ctx=1024,
            keep_alive=120, client_kwargs={"timeout": 180, "trust_env": False},
        )
        self.database = Chroma(
            collection_name="campus_support", persist_directory=str(folder), embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )

    def existing_ids(self):
        rows = self.database.get(include=["documents", "metadatas"])
        ids, texts, metadata = rows["ids"], rows["documents"], rows["metadatas"]
        if len(ids) != len(set(ids)) or len(ids) != len(texts) or len(ids) != len(metadata):
            raise ValueError("Inconsistent index records")
        for key, text, fields in zip(ids, texts, metadata):
            original = self.sources.get(key)
            if original is None or text != original.page_content or fields != original.metadata:
                raise ValueError(f"Indexed content differs from the original: {key}")
        return set(ids)

    def build(self):
        present = self.existing_ids()
        missing = [doc for doc in self.documents if doc.metadata["policy_id"] not in present]
        for start in range(0, len(missing), 16):
            wait_for_gpu()
            batch = missing[start:start + 16]
            self.database.add_documents(batch, ids=[doc.metadata["policy_id"] for doc in batch])
            print(f"Stored {len(present) + min(start + 16, len(missing))}/{len(self.documents)}", flush=True)
        if self.existing_ids() != set(self.sources):
            raise RuntimeError("The index is incomplete")

    def retrieve(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("Enter a question containing 1 to 1000 characters")
        if self.existing_ids() != set(self.sources):
            raise RuntimeError("Build the policy index before asking questions")
        wait_for_gpu()
        hits = self.database.similarity_search(question, k=2)
        if not hits:
            raise RuntimeError("Retrieval returned no evidence")
        for doc in hits:
            expected = self.sources.get(doc.metadata.get("policy_id"))
            if expected is None or doc.page_content != expected.page_content or doc.metadata != expected.metadata:
                raise ValueError("Retrieved evidence does not match the supplied policy")
        return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--index", type=Path, default=HOME / ".index")
    args = parser.parse_args()
    PolicyIndex(args.index, args.base_url).build()
