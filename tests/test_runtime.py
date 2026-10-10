import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag import PolicyIndex
from thermal_guard import wait_for_gpu


class ThermalTests(unittest.TestCase):
    def test_disabled_guard_does_not_touch_gpu(self):
        with patch.dict(os.environ, {"POLICY_THERMAL_GUARD": "0"}), patch("thermal_guard.subprocess.run") as run:
            wait_for_gpu()
            run.assert_not_called()

    def test_hard_limit(self):
        with patch.dict(os.environ, {"POLICY_THERMAL_GUARD": "1"}), patch(
            "thermal_guard.subprocess.run", return_value=Mock(stdout="80")
        ):
            with self.assertRaises(RuntimeError):
                wait_for_gpu()

    def test_cooling_until_safe(self):
        with patch.dict(os.environ, {"POLICY_THERMAL_GUARD": "1"}), patch(
            "thermal_guard.subprocess.run", side_effect=[Mock(stdout="73"), Mock(stdout="68"), Mock(stdout="65")]
        ), patch("thermal_guard.time.sleep") as sleep:
            wait_for_gpu()
            self.assertEqual(sleep.call_count, 2)

    def test_monitor_error_not_ignored(self):
        with patch.dict(os.environ, {"POLICY_THERMAL_GUARD": "1"}), patch(
            "thermal_guard.subprocess.run", side_effect=OSError("no monitor")
        ):
            with self.assertRaises(OSError):
                wait_for_gpu()


class ChromaTests(unittest.TestCase):
    def test_build_and_read_real_chroma(self):
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_core.embeddings import Embeddings

        class FixedEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[float(len(text)), 1.0, 0.5] for text in texts]

            def embed_query(self, text):
                return [float(len(text)), 1.0, 0.5]

        docs = [
            Document(page_content="Print jobs expire after 24 hours.", metadata={
                "policy_id": "doc-1", "topic": "printing", "source": "university_it_policies.txt"}),
            Document(page_content="Visitors use Campus-Guest.", metadata={
                "policy_id": "doc-2", "topic": "wifi", "source": "university_it_policies.txt"}),
        ]
        store = Chroma(collection_name="offline_policy_test", embedding_function=FixedEmbeddings(),
                       client_settings=Settings(anonymized_telemetry=False))
        try:
            with patch("rag.load_policy_documents", return_value=docs):
                index = PolicyIndex(database=store)
            index.build()
            self.assertEqual(index.existing_ids(), {"doc-1", "doc-2"})
            self.assertEqual(len(index.retrieve("printing")), 2)
            store.update_documents(["doc-1"], [Document(page_content="Changed", metadata=docs[0].metadata)])
            with self.assertRaises(ValueError):
                index.existing_ids()
        finally:
            store.delete_collection()


if __name__ == "__main__":
    unittest.main()
