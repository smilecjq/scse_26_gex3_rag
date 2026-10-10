# Campus IT Policy Assistant

This is a local question-answering program for the university IT policies supplied with SCSE GEX3. It searches those policies before answering, and checks the answer against the retrieved text. It has no web search or separate contact directory.

## Running It

Use Python 3.12 and install Ollama. From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

With Ollama running, build the index once and ask a question:

```powershell
.\.venv\Scripts\python.exe rag.py
.\.venv\Scripts\python.exe agent.py "What exact phone number should I call for the University IT Service Desk?" --record runs/phone.json
```

Answers are printed to the terminal. The record file also contains the retrieved passages, model messages, evidence assessment, review and any revision. Choose a different record path for each run.

The models run locally. The configuration requests GPU offloading and limits CPU math to two threads. Without a compatible GPU, Ollama may run more slowly on the CPU; the verification script explicitly checks full GPU offloading for the recorded local validation.

## From Question To Answer

`policy_loader.py` is the original course loader and is unchanged. It reads the single `university_it_policies.txt` source. `rag.py` embeds its policy sections using `nomic-embed-text` and keeps them in Chroma. The source digest and stored text are checked before an index is reused; an interrupted build can continue with the missing sections.

`rag_tool.py` exposes retrieval as a LangChain tool. It returns two passages with their policy IDs and source names. `agent.py` then asks Qwen3-0.6B whether the passages contain the requested fact, drafts an answer and reviews it. A failed draft gets one revision followed by another review. The supplied course corpus is not used as a second knowledge source.

A supported answer needs an exact quote from a retrieved policy. Local checks reject invented citations, altered quotes and numerical claims that do not match the quote. The model reviewer checks the meaning of the answer. For an unknown answer, a separate model call checks whether the requested fact is actually absent; the program converts that finding into PASS or REVISE. That prevents a blanket unknown answer from passing when the policy supplies the fact.

When evidence is missing, the response must be only "I don't know." A failed model request, an incomplete index or malformed output is an error, not a substitute unknown answer. Passage headings are omitted from model prompts to discourage title-only citations; the index and saved retrieval records retain the full original text.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe verify_live.py --record runs/check.json
```

The unit tests use controlled responses to exercise the program logic. They do not establish model quality. The live check uses the actual local model for the phone question, printing retention, visitor Wi-Fi and account lock duration. It also asks the reviewer to reject a wrong network name and an unjustified unknown answer.

On 10 October 2026, all 37 unit tests and two complete live checks passed. The original live records are in `validation/live-check.json` and `validation/repeat-check.json`. Both runs returned a bare "I don't know" for the phone question with no quote or policy ID; the other answers were 24 hours, Campus-Guest and 15 minutes. Both incorrect-answer controls were rejected. These records include the model messages and GPU offloading checks.

An earlier run rejected an unknown answer because it still carried an unrelated policy ID. The field descriptions and drafting prompt were clarified before the two recorded checks; the answer text was not edited after generation. Local failed-run logs remain in `runs/`.

This is still a small-model experiment. Retrieval can miss relevant text, and an automated reviewer can make mistakes. Passing a fixed set of questions does not establish that every possible answer is correct.

## Laptop Runs

The optional PowerShell 7 wrapper runs a temporary local Ollama service, monitors available RAM and GPU temperature, and closes its own processes afterwards:

```powershell
pwsh -File run_local.ps1 -Python .venv/Scripts/python.exe -Mode build
pwsh -File run_local.ps1 -Python .venv/Scripts/python.exe -Mode verify
```

The wrapper requires 4.5 GiB of free RAM and 4500 MiB of free GPU memory before starting. It pauses model work at 72 C until the GPU cools to 65 C, and stops at 80 C or below 1.25 GiB of free RAM. These are software checks, not a guarantee against hardware problems. `-Backend vulkan -VulkanDevice 1` is available for machines where Ollama's CUDA backend cannot use the installed driver; the device index is machine-specific.

Model downloads, the local index, environments and temporary runs are excluded from Git. Keep the generated records when reviewing an answer or investigating a failed run.
