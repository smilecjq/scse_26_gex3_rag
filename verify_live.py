import argparse
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from agent import MODEL, Answer, create_assistant, unknown

QUESTIONS = {
    "phone": "What exact phone number should I call for the University IT Service Desk?",
    "printing": "How long are unreleased print jobs retained?",
    "wifi": "Which wireless network should visitors use?",
    "account": "How long does a temporary account lock last after repeated failed logins?",
}


def matches(case, answer):
    if case == "phone":
        return unknown(answer)
    if case == "wifi":
        return "Campus-Guest" in answer and "Campus-Secure" not in answer
    number, unit = ("24", "hours?") if case == "printing" else ("15", "minutes?")
    return bool(re.search(r"\b" + number + r"\s+" + unit + r"\b", answer, re.I)) and set(re.findall(r"\d+", answer)) == {number}


def gpu_state(endpoint, name):
    with build_opener(ProxyHandler({})).open(endpoint + "/api/ps", timeout=10) as response:
        snapshot = json.load(response)
    tag = name if ":" in name else name + ":latest"
    rows = [item for item in snapshot.get("models", []) if item["name"] == tag]
    if not rows or any(row.get("size", 0) <= 0 or row.get("size_vram", 0) < row["size"] for row in rows):
        raise RuntimeError("Full GPU offloading was not reported for " + name)
    return rows


def run(endpoint, folder):
    assistant = create_assistant(endpoint, folder)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
              "versions": {p: importlib.metadata.version(p) for p in
                           ("langchain-ollama", "langchain-chroma", "chromadb")},
              "questions": [], "controls": [], "gpu": []}
    for case, question in QUESTIONS.items():
        try:
            assistant.retriever.invoke({"question": question})
            record["gpu"].append(gpu_state(endpoint, "nomic-embed-text"))
            answer = assistant.ask(question)
            record["gpu"].append(gpu_state(endpoint, MODEL))
            answer.update(case=case, passed=matches(case, answer["answer"]))
        except Exception as error:
            answer = {"case": case, "passed": False, "error": str(error), "trace": getattr(error, "trace", [])}
        record["questions"].append(answer)
        print(case, "PASS" if answer["passed"] else "FAIL", flush=True)
    for case in ("wifi", "printing"):
        trace = []
        try:
            question = QUESTIONS[case]
            passages = assistant.retriever.invoke({"question": question})
            assessment = assistant.assess(question, passages, trace)
            if case == "wifi":
                policy = next(p for p in passages if "Campus-Guest" in p["text"])
                candidate = Answer(answer="Use Campus-Secure.", quote=policy["text"], policy_id=policy["id"])
            else:
                candidate = Answer(answer="I don't know.", quote="", policy_id="")
            verdict, errors = assistant.reflect(question, candidate, passages, assessment, trace)
            control = {"case": case, "candidate": candidate.model_dump(), "review": verdict.model_dump(),
                       "errors": errors, "passed": verdict.verdict == "REVISE", "trace": trace}
        except Exception as error:
            control = {"case": case, "passed": False, "error": str(error), "trace": trace}
        record["controls"].append(control)
        print("control", case, "PASS" if control["passed"] else "FAIL", flush=True)
    record["passed"] = all(row["passed"] for row in record["questions"] + record["controls"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--index", type=Path, default=Path(__file__).resolve().parent / ".index")
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.record.exists():
        parser.error("Use a fresh output path")
    try:
        report = run(args.base_url, args.index)
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    raise SystemExit(0 if report["passed"] else 1)
