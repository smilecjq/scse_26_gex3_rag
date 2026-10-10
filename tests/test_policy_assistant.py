import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from pydantic import ValidationError

from agent import Answer, Assessment, PolicyAssistant, UnverifiedAnswer, Verdict, evidence_errors, unknown
from rag import PolicyIndex
from rag_tool import make_retriever
from verify_live import matches

TEXT = "Printing\n\nUnreleased print jobs are retained for 24 hours."
PASSAGE = {"id": "doc-1", "text": TEXT, "topic": "printing", "source": "university_it_policies.txt"}
GOOD = Answer(answer="24 hours", quote="Unreleased print jobs are retained for 24 hours.", policy_id="doc-1")
UNKNOWN = Answer(answer="I don't know.", quote="", policy_id="")
FOUND = Assessment(reason="The policy states 24 hours.", fact_found=True)
ABSENT = Assessment(reason="The requested fact is not stated.", fact_found=False)
PASS = Verdict(reason="The answer agrees with the policy.", verdict="PASS")
REVISE = Verdict(reason="The candidate gives a different fact.", verdict="REVISE")


class EvidenceTests(unittest.TestCase):
    def test_live_answer_checks(self):
        self.assertTrue(matches("phone", "I don't know."))
        self.assertFalse(matches("phone", "I don't know. Call reception."))
        self.assertTrue(matches("printing", "24 hours"))
        self.assertFalse(matches("printing", "24 days"))
        self.assertTrue(matches("wifi", "Campus-Guest"))
        self.assertFalse(matches("wifi", "Campus-Secure"))
        self.assertTrue(matches("account", "15 minutes"))
        self.assertFalse(matches("account", "15 hours"))

    def test_good_quote(self):
        self.assertEqual(evidence_errors(GOOD, [PASSAGE]), [])

    def test_missing_policy(self):
        self.assertTrue(evidence_errors(GOOD.model_copy(update={"policy_id": "doc-999"}), [PASSAGE]))

    def test_fabricated_quote(self):
        self.assertTrue(evidence_errors(GOOD.model_copy(update={"quote": "Made up text"}), [PASSAGE]))

    def test_title_is_not_evidence(self):
        self.assertTrue(evidence_errors(GOOD.model_copy(update={"quote": "Printing"}), [PASSAGE]))

    def test_new_number(self):
        self.assertTrue(evidence_errors(GOOD.model_copy(update={"answer": "48 hours"}), [PASSAGE]))

    def test_wrong_unit(self):
        self.assertTrue(evidence_errors(GOOD.model_copy(update={"answer": "24 days"}), [PASSAGE]))

    def test_unknown_is_bare(self):
        self.assertTrue(unknown("I don't know"))
        self.assertFalse(unknown("I don't know. Contact support."))

    def test_unknown_has_no_citation(self):
        self.assertEqual(evidence_errors(UNKNOWN, [PASSAGE]), [])
        self.assertTrue(evidence_errors(UNKNOWN.model_copy(update={"policy_id": "doc-1"}), [PASSAGE]))

    def test_empty_answer_rejected(self):
        with self.assertRaises(ValidationError):
            Answer(answer="", quote="", policy_id="")

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Answer(answer="test", quote="", policy_id="", extra="data")


class WorkflowTests(unittest.TestCase):
    def assistant(self, replies, passages=None):
        model = Mock()
        model.request.side_effect = replies
        retriever = Mock()
        retriever.invoke.return_value = [PASSAGE] if passages is None else passages
        return PolicyAssistant(retriever, model)

    def test_known_answer(self):
        worker = self.assistant([FOUND, GOOD, PASS])
        self.assertEqual(worker.ask("How long?")["answer"], "24 hours")
        self.assertEqual(worker.model.request.call_count, 3)

    def test_unknown_uses_independent_assessment(self):
        worker = self.assistant([ABSENT, UNKNOWN, ABSENT])
        result = worker.ask("What is the phone number?")
        self.assertEqual(result["answer"], "I don't know.")
        self.assertEqual(result["trace"][-1]["method"], "independent evidence assessment")

    def test_wrong_answer_is_revised(self):
        bad = GOOD.model_copy(update={"answer": "48 hours"})
        worker = self.assistant([FOUND, bad, REVISE, GOOD, PASS])
        self.assertEqual(worker.ask("How long?")["revisions"], 1)

    def test_false_unknown_is_revised(self):
        worker = self.assistant([FOUND, UNKNOWN, FOUND, GOOD, PASS])
        self.assertEqual(worker.ask("How long?")["answer"], "24 hours")

    def test_model_pass_cannot_override_evidence_errors(self):
        bad = GOOD.model_copy(update={"answer": "48 hours"})
        with self.assertRaises(UnverifiedAnswer):
            self.assistant([FOUND, bad, PASS, bad, PASS]).ask("How long?")

    def test_missing_fact_cannot_produce_known_answer(self):
        with self.assertRaises(UnverifiedAnswer):
            self.assistant([ABSENT, GOOD, PASS, GOOD, PASS]).ask("A missing fact?")

    def test_no_evidence_is_not_unknown(self):
        with self.assertRaises(RuntimeError):
            self.assistant([], []).ask("Phone?")

    def test_service_failure_is_not_unknown(self):
        worker = self.assistant([])
        worker.model.request.side_effect = ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            worker.ask("Phone?")

    def test_invalid_questions(self):
        worker = self.assistant([])
        for value in ("", " ", "a" * 1001, None):
            with self.subTest(value=str(value)[:10]), self.assertRaises(ValueError):
                worker.ask(value)
        worker.retriever.invoke.assert_not_called()

    def test_assessment_does_not_see_candidate(self):
        worker = self.assistant([ABSENT, UNKNOWN, ABSENT])
        worker.ask("Phone?")
        payload = worker.model.request.call_args_list[0].args[2]
        self.assertEqual(set(payload), {"question", "evidence"})

    def test_review_is_required(self):
        with self.assertRaises(UnverifiedAnswer):
            self.assistant([FOUND, GOOD, REVISE, GOOD, REVISE]).ask("How long?")


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(page_content=TEXT, metadata={
            "policy_id": "doc-1", "topic": "printing", "source": "university_it_policies.txt"})
        self.store = Mock()
        self.store.get.return_value = {"ids": ["doc-1"], "documents": [TEXT], "metadatas": [self.document.metadata]}
        self.store.similarity_search.return_value = [self.document]
        with patch("rag.load_policy_documents", return_value=[self.document]):
            self.index = PolicyIndex(database=self.store)

    def test_uses_original_loader(self):
        self.assertEqual(len(self.index.documents), 1)

    def test_exact_retrieval(self):
        self.assertEqual(self.index.retrieve("Printing?"), [self.document])

    def test_incomplete_index(self):
        self.store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with self.assertRaises(RuntimeError):
            self.index.retrieve("Printing?")

    def test_altered_text_rejected(self):
        self.store.get.return_value["documents"] = ["changed"]
        with self.assertRaises(ValueError):
            self.index.existing_ids()

    def test_altered_metadata_rejected(self):
        self.store.get.return_value["metadatas"] = [{"policy_id": "doc-1"}]
        with self.assertRaises(ValueError):
            self.index.existing_ids()

    def test_empty_hits_rejected(self):
        self.store.similarity_search.return_value = []
        with self.assertRaises(RuntimeError):
            self.index.retrieve("Printing?")

    def test_invalid_hit_rejected(self):
        self.store.similarity_search.return_value = [Document(page_content="Other", metadata=self.document.metadata)]
        with self.assertRaises(ValueError):
            self.index.retrieve("Printing?")

    def test_completed_build_does_not_reembed(self):
        self.index.build()
        self.store.add_documents.assert_not_called()

    def test_resume_checks_inserted_records(self):
        complete = self.store.get.return_value
        self.store.get.side_effect = [{"ids": [], "documents": [], "metadatas": []}, complete]
        self.index.build()
        self.store.add_documents.assert_called_once_with([self.document], ids=["doc-1"])

    def test_tool_returns_policy_provenance(self):
        rows = make_retriever(self.index).invoke({"question": "Printing?"})
        self.assertEqual(rows, [PASSAGE])


if __name__ == "__main__":
    unittest.main()
