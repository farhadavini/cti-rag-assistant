#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import yaml

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pypdf nicht installiert, PDF-Support deaktiviert")

# Embedding Providers
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()

# Logging: Details in Datei, nur Fehler auf Console
file_handler = logging.FileHandler("ingest.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# FAISS Warnings unterdrücken
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Python warnings für FAISS unterdrücken
import warnings
warnings.filterwarnings("ignore", message=".*AVX2.*")
warnings.filterwarnings("ignore", module="faiss")


def load_config():
    """Config laden und prüfen."""
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yaml nicht gefunden")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML-Format ungültig: {e}")
        raise

    # Prüfe ob alle Sections da sind
    required = ["data", "chunking", "embedding", "retrieval", "llm"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config fehlt: {key}")

    if cfg["chunking"]["size"] <= 0:
        raise ValueError("chunking.size muss > 0 sein")

    if cfg["chunking"]["overlap"] < 0:
        raise ValueError("chunking.overlap muss >= 0 sein")

    return cfg


def load_pdf(filepath):
    """PDF Datei laden und Text extrahieren."""
    if not PDF_AVAILABLE:
        return None

    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            # extract_text() kann bei Seiten ohne Textlayer None liefern.
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        logger.warning(f"PDF fehler {filepath.name}: {e}")
        return None


def load_json(filepath):
    """JSON Datei laden - nimmt 'text' oder 'content' Feld."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Versuche verschiedene gängige Felder
        if isinstance(data, dict):
            text = data.get('text') or data.get('content') or data.get('body')
            if text:
                return text
            # Falls kein Text-Feld, ganzes JSON als String
            return json.dumps(data, ensure_ascii=False)
        elif isinstance(data, list):
            # Liste von Texten oder Objekten
            texts = []
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    t = item.get('text') or item.get('content') or json.dumps(item)
                    texts.append(t)
            return "\n".join(texts)
        else:
            return str(data)
    except Exception as e:
        logger.warning(f"JSON fehler {filepath.name}: {e}")
        return None


def load_documents(folder):
    """Alle Dokumente (TXT, PDF, JSON) aus Ordner laden. Bei Duplikaten .txt bevorzugen."""
    folder_path = Path(folder)

    # Alle Dateien nach Typ sammeln
    txt_files = list(folder_path.glob("*.txt"))
    pdf_files = list(folder_path.glob("*.pdf")) if PDF_AVAILABLE else []
    json_files = list(folder_path.glob("*.json"))

    # Nach Basename (ohne Extension) gruppieren um Duplikate zu finden
    files_by_basename = {}

    for f in txt_files:
        basename = f.stem
        if basename not in files_by_basename:
            files_by_basename[basename] = {'txt': f}
        else:
            files_by_basename[basename]['txt'] = f

    for f in pdf_files:
        basename = f.stem
        if basename not in files_by_basename:
            files_by_basename[basename] = {'pdf': f}
        else:
            files_by_basename[basename]['pdf'] = f

    for f in json_files:
        basename = f.stem
        if basename not in files_by_basename:
            files_by_basename[basename] = {'json': f}
        else:
            files_by_basename[basename]['json'] = f

    # Für jeden Basename beste Datei wählen (Präferenz: txt > pdf > json)
    docs = []
    total_files = len(files_by_basename)

    for i, (basename, formats) in enumerate(files_by_basename.items(), 1):
        text = None
        chosen_file = None
        file_type = None

        # Präferenz-Reihenfolge: txt > pdf > json
        if 'txt' in formats:
            try:
                text = formats['txt'].read_text(encoding="utf-8", errors='ignore').strip()
                chosen_file = formats['txt']
                file_type = 'txt'
            except Exception as e:
                logger.warning(f"fehler {formats['txt'].name}: {e}")

        if not text and 'pdf' in formats:
            text = load_pdf(formats['pdf'])
            if text:
                chosen_file = formats['pdf']
                file_type = 'pdf'

        if not text and 'json' in formats:
            text = load_json(formats['json'])
            if text:
                chosen_file = formats['json']
                file_type = 'json'

        # Nur Dokumente mit Inhalt speichern - Nur mehr als 50 
        if text and len(text) > 50:
            docs.append({
                'content': text,
                'filename': chosen_file.name,
                'filepath': str(chosen_file),
                'type': file_type
            })

        if i % 100 == 0:
            logger.info(f"{i}/{total_files} dateien")

    logger.info(f"{len(docs)} dokumente geladen (txt={sum(1 for d in docs if d['type']=='txt')}, pdf={sum(1 for d in docs if d['type']=='pdf')}, json={sum(1 for d in docs if d['type']=='json')})")
    return docs


def chunk_documents(docs, size, overlap):
    """Dokumente in kleinere Chunks aufteilen."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )

    chunks = []
    for doc in docs:
        for idx, chunk_text in enumerate(splitter.split_text(doc['content'])):
            chunks.append({
                'content': chunk_text,
                'filename': doc['filename'],
                'filepath': doc['filepath'],
                'chunk_index': idx
            })

    logger.info(f"{len(chunks)} chunks")
    return chunks


def create_embeddings(config, api_key, task_type="retrieval_document"):
    """Embedding Model- Providerabhängig."""
    provider = config.get("embedding", {}).get("provider", "google").lower()
    model = config["embedding"]["model"]

    if provider == "google":
        if not GOOGLE_AVAILABLE:
            raise ImportError("langchain-google-genai nicht installiert")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
            task_type=task_type
        )
    elif provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai nicht installiert")
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key
        )
    else:
        raise ValueError(f"Provider '{provider}' nicht unterstützt. Nutze: google, openai")


def embed_documents(doc_path: str):
    """Dokumente laden, chunken und embedden. Speichert FAISS Index."""
    # Config und API-Key
    try:
        cfg = load_config()
    except Exception as e:
        logger.error(f"Config-Fehler: {e}")
        return False

    batch_size = cfg.get("ingestion", {}).get("batch_size", 1000)

    api_key_env = cfg.get("embedding", {}).get("api_key_env", cfg["llm"]["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.error(f"key fehlt: {api_key_env}")
        return False

    # Dokumente laden und chunken
    docs = load_documents(doc_path)
    if not docs:
        logger.error("keine dokumente")
        return False

    chunks = chunk_documents(docs, cfg["chunking"]["size"], cfg["chunking"]["overlap"])

    # LangChain Documents erstellen mit Metadaten
    documents = [
        Document(
            page_content=c['content'],
            metadata={
                "chunk_id": i,
                "source_file": c['filename'],
                "filepath": c['filepath'],
                "chunk_index": c['chunk_index']
            }
        ) for i, c in enumerate(chunks)
    ]

    # Embedding Model initialisieren
    try:
        embeddings = create_embeddings(cfg, api_key, task_type="retrieval_document")
    except Exception as e:
        logger.error(f"Embedding-Initialisierung fehlgeschlagen: {e}")
        return False

    # FAISS Index in Batches erstellen
    logger.info(f"vectorstore erstellen...")
    vectorstore = None

    try:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

        vectorstore.save_local(cfg["data"]["index_path"])
        logger.info(f"gespeichert: {cfg['data']['index_path']}")
    except Exception as e:
        logger.error(f"Vectorstore-Fehler: {e}")
        return False

    # Metadaten speichern
    try:
        with open(cfg["data"]["metadata_file"], "w") as f:
            json.dump({
                "embedding_model": cfg["embedding"]["model"],
                "chunk_size": cfg["chunking"]["size"],
                "chunk_overlap": cfg["chunking"]["overlap"],
                "retrieval_k": cfg["retrieval"]["k"],
                "num_documents": len(docs),
                "num_chunks": len(chunks)
            }, f, indent=2)
    except IOError as e:
        logger.error(f"Metadata speichern fehlgeschlagen: {e}")
        return False

    logger.info("fertig")
    return True


if __name__ == "__main__":
    try:
        cfg = load_config()
        success = embed_documents(cfg["data"]["folder"])
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fehler: {e}")
        sys.exit(1)
