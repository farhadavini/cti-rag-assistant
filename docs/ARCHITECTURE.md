# Architecture

This document describes the retrieval-augmented generation (RAG) pipeline, its
modules and configuration. The README covers setup and usage.

## 1. Overview

The assistant answers questions about a local corpus of Cyber Threat
Intelligence (CTI) reports. It is strictly grounded: the language model only
sees passages retrieved from the corpus and is instructed to answer from that
context alone, with source attribution. There are two entry points that share
the same core: an interactive CLI (`query.py`) and a minimal Flask web UI
(`web.py`).

## 2. Two phases

### Ingestion (offline, build the index)

```
data/*.txt|pdf|json
   -> load_documents()      pick best format per report (txt > pdf > json)
   -> chunk_documents()     RecursiveCharacterTextSplitter (size + overlap)
   -> Document(page_content, metadata={source_file, chunk_index, ...})
   -> create_embeddings()   provider from config (google | openai)
   -> FAISS.from_documents  built in batches
   -> save_local(faiss_index/) + metadata.json
```

### Query (online, answer a question)

```
question
   -> get_embeddings() + get_vectorstore()  load faiss_index/
   -> retriever        top-k similarity (optional score threshold)
   -> RetrievalQA      strict context-only prompt, chain_type from config
   -> create_llm()     provider from config (google | openai)
   -> answer_text + source_documents
   -> _format_response()  answer + de-duplicated source list
```

## 3. Modules

| Module | Responsibility |
|---|---|
| `ingest.py` | Load documents (TXT/PDF/JSON), chunk them, embed them and persist the FAISS index and `metadata.json`. Handles PDF text extraction defensively (`extract_text() or ""`). |
| `query.py` | Config loading and validation, embedding/LLM/vector-store construction, the RAG chain, and `answer()`. Builds the chain once and caches it (`_get_chain`) so the index, embeddings and LLM are not reloaded per question. |
| `web.py` | Single-page Flask UI; delegates all work to `query.answer()`. |
| `eval.py` | Offline evaluation harness over a set of questions (`eval_questions.example.json`). |
| `config.yaml` | All tunables (see section 5). |
| `examples/` | A few example CTI reports so the pipeline can be demoed without sourcing your own. |

## 4. Provider abstraction

Both the embedding model and the chat model are provider-selectable, with
parity between them:

- `create_embeddings(config, api_key, task_type)` returns a Google
  (`GoogleGenerativeAIEmbeddings`) or OpenAI (`OpenAIEmbeddings`) embedder.
- `create_llm(config)` returns a Google (`ChatGoogleGenerativeAI`) or OpenAI
  (`ChatOpenAI`) chat model.

The provider is chosen per section in `config.yaml` (`embedding.provider`,
`llm.provider`). The code imports the LangChain split packages directly
(`langchain_core`, `langchain_community`, `langchain_classic`,
`langchain_text_splitters`, provider packages), not the `langchain`
meta-package.

## 5. Configuration (`config.yaml`)

- `data.folder` / `index_path` / `metadata_file`: corpus and index locations.
- `chunking.size` / `overlap`: chunk length and overlap in characters.
- `embedding.provider` / `model` (+ optional `api_key_env`).
- `retrieval.k`: number of chunks passed to the LLM.
- `retrieval.use_threshold` / `threshold`: optional similarity cut-off.
- `llm.provider` / `model` / `temperature` / `max_tokens` / `chain_type`
  (`stuff`, `map_reduce`, `refine`, `map_rerank`).
- `ingestion.batch_size`: documents per embedding batch.

## 6. Grounding and failure behaviour

- The prompt forbids answers outside the retrieved context and requires a fixed
  "no information found" response when nothing matches.
- `answer()` treats a result with no retrieved source documents as "no answer",
  independent of the prompt language (it checks for sources, not a fixed
  sentence).
- Common API errors (quota / 429 / resource exhausted) are mapped to a friendly
  message instead of surfacing a stack trace.

## 7. Security notes

- `FAISS.load_local(..., allow_dangerous_deserialization=True)` is used because
  the index is pickle-based. This is safe for a self-built local index; never
  load a FAISS index from an untrusted source.
- API keys are read from the environment (`.env` via python-dotenv); none are
  stored in the repository.
