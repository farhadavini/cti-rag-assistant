# CTI RAG Assistant

![CI](../../actions/workflows/ci.yml/badge.svg) ![Code: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg) ![Docs: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)

Retrieval-augmented question answering over Cyber Threat Intelligence reports.
Documents (TXT, PDF, JSON) are chunked, embedded and stored in a local FAISS
index; questions are answered by an LLM strictly from the retrieved passages,
with source attribution.

**Stack:** LangChain | FAISS | Google Gemini or OpenAI (embeddings **and** LLM, each selectable via `config.yaml`)

> **Data & licensing.** This repository ships **only synthetic example reports**
> (`examples/`, fictional, marked `TLP:CLEAR`) so the pipeline can be tried without
> any third-party data. **No real, copyrighted threat-intelligence reports and no
> pre-built index are included.** To run it on real data, add your own reports to
> `data/` (or obtain a public corpus such as the [ORKL library](https://orkl.eu)
> under its terms) and build the index locally with `python ingest.py`. API keys
> stay in your local `.env` and are never committed.

Detailed design (phases, modules, provider abstraction): see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## How it works

```
data/*.txt|pdf|json --> ingest.py --> chunks --> embeddings --> faiss_index/
                                                                     |
question --> query.py --> embed --> top-k similarity search ---------+
                             +--> prompt (context only) --> LLM --> answer + sources
```

- If the same report exists in several formats, `.txt` wins over `.pdf` over `.json`.
- The prompt forbids answers outside the retrieved context; without matching
  context the assistant says it has no information instead of guessing.
- Answers are given in the language of the question (German or English).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then insert your own API key
```

## Usage

1. Put your CTI reports into `data/` (`.txt`, `.pdf` or `.json`). To try it
   without sourcing your own, copy the bundled examples:

   ```bash
   cp examples/*.txt data/
   ```

2. Build the index:

   ```bash
   python ingest.py
   ```

   Creates `faiss_index/` and `metadata.json`.

3. Ask questions interactively:

   ```bash
   python query.py
   ```

   or from Python:

   ```python
   from query import answer

   result = answer("Which initial access techniques does APT29 use?", include_sources=True)
   print(result["answer"])
   for s in result["sources"]:
       print("-", s["file"])
   ```

4. Optional web UI (`pip install flask`):

   ```bash
   python web.py            # http://127.0.0.1:5001
   ```

## Evaluation

Measure retrieval/answer quality against a fixed question set instead of
eyeballing it:

```bash
python eval.py eval_questions.example.json
```

Each case passes if the answer contains its expected keywords (or, without
any, if sources were retrieved); the script reports pass rate and source
coverage.

## Configuration (`config.yaml`)

| Key | Meaning |
|---|---|
| `data.folder` | input documents (default `data`) |
| `chunking.size` / `overlap` | chunk size and overlap in characters |
| `embedding.provider` / `model` | `google` or `openai` and the embedding model |
| `retrieval.k` | number of chunks passed to the LLM |
| `retrieval.use_threshold` / `threshold` | optional similarity cut-off |
| `llm.provider` / `model` | `google` (Gemini) or `openai` (GPT) and the chat model |
| `llm.temperature` / `max_tokens` | generation settings |
| `llm.chain_type` | `stuff` (recommended), `map_reduce`, `refine`, `map_rerank` |
| `ingestion.batch_size` | documents per embedding batch |

Model names change over time; set `embedding.model` and `llm.model` to models
currently offered by your provider. To use OpenAI for embeddings and/or the LLM,
install `langchain-openai`, set the provider(s) to `openai` and set
`OPENAI_API_KEY`.

## Notes

- API keys live only in `.env`, which is git-ignored.
- `faiss_index/` is loaded with `allow_dangerous_deserialization=True`; only
  load indices you created yourself.
- Logs are written to `ingest.log` and `query.log`; the console shows errors only.
- On the Gemini free tier, rate limits apply; the assistant reports when the
  quota is exhausted.

## License

- **Code:** MIT, see [LICENSE](LICENSE).
- **Documentation** (this README and `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), see [LICENSE-docs](LICENSE-docs).
  Reuse is welcome with attribution: *Farhad Yaghoubi Avini, https://farhadavini.com*.
