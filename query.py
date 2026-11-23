#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from dotenv import load_dotenv
import yaml
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Embedding und LLM Providers
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()

# Logging: Details in Datei, nur Fehler auf Console
file_handler = logging.FileHandler("query.log")
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

# LangChain und FAISS Warnings unterdrücken - nicht im CLI sichtbar werden
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Zusätzlich Python warnings für FAISS unterdrücken - Falls Cpu AVX2 nicht unterstützt und system Funktionsfähig bleibt
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

    # Prüfe ob alle wichtige benötigten Sections da sind
    required = ["data", "chunking", "embedding", "retrieval", "llm"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config fehlt: {key}")

    # Validierung wichtiger Parameter
    if cfg["retrieval"]["k"] <= 0:
        raise ValueError("retrieval.k muss > 0 sein")

    if cfg["chunking"]["size"] <= 0:
        raise ValueError("chunking.size muss > 0 sein")

    return cfg


def create_embeddings(config, api_key, task_type="retrieval_query"):  # OpenAI / Gemini
    #Embedding Model erstellen je nach Provider.
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


def get_embeddings(config): 
    key_env = config.get("embedding", {}).get("api_key_env", config["llm"]["api_key_env"]) # optional einstelbar - Seperate schlüssel für rag und Embding
    key = os.getenv(key_env)
    if not key:
        logger.error("embedding key fehlt")
        return None

    try:
        return create_embeddings(config, key, task_type="retrieval_query")
    except Exception as e:
        logger.error(f"Embedding-Initialisierung fehlgeschlagen: {e}")
        return None


def create_llm(config):
    """Erstellt das Chat-LLM je nach Provider (Parität zu create_embeddings).

    Provider aus config['llm']['provider'] (Default 'google'). Unterstützt
    'google' (Gemini) und 'openai' (GPT). API-Key aus dem konfigurierten
    api_key_env.
    """
    llm_cfg = config["llm"]
    provider = llm_cfg.get("provider", "google").lower()
    api_key = os.getenv(llm_cfg["api_key_env"])
    if not api_key:
        logger.error(f"LLM API Key fehlt: {llm_cfg['api_key_env']}")
        return None

    try:
        if provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("langchain-google-genai nicht installiert")
            return ChatGoogleGenerativeAI(
                model=llm_cfg["model"],
                google_api_key=api_key,
                temperature=llm_cfg["temperature"],
                max_tokens=llm_cfg["max_tokens"],
            )
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("langchain-openai nicht installiert")
            return ChatOpenAI(
                model=llm_cfg["model"],
                openai_api_key=api_key,
                temperature=llm_cfg["temperature"],
                max_tokens=llm_cfg["max_tokens"],
            )
        else:
            raise ValueError(f"LLM-Provider '{provider}' nicht unterstützt. Nutze: google, openai")
    except Exception as e:
        logger.error(f"LLM-Initialisierung fehlgeschlagen: {e}")
        return None


def get_vectorstore(config, embeddings):
    """FAISS Index laden."""
    from pathlib import Path

    index_path = Path(config["data"]["index_path"])
    if not index_path.exists():
        logger.error(f"index nicht gefunden: {index_path}")
        return None

    try:
        # FAISS Index laden - allow_dangerous_deserialization nur für lokale Indices!
        vectorstore = FAISS.load_local(
            config["data"]["index_path"],
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Prüfe ob Index Vektoren enthält
        if vectorstore.index.ntotal == 0:
            logger.error("index ist leer")
            return None

        logger.info(f"index geladen: {vectorstore.index.ntotal} vektoren")
        return vectorstore

    except Exception as e:
        logger.error(f"fehler beim laden: {e}")
        return None


def create_rag_chain(config, vectorstore):
    """RAG Chain erstellen """
    llm = create_llm(config)
    if llm is None:
        return None

    # Retrieval konfigurieren - FLEXIBILITÄT für Threshold + k (Einstelbar durch config.yaml)
    retrieval = config["retrieval"]
    if retrieval.get("use_threshold", False):
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": retrieval["k"],
                "score_threshold": retrieval["threshold"]
            }
        )
        logger.info(f"Retrieval: k={retrieval['k']} + threshold={retrieval['threshold']}")
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": retrieval["k"]}
        )
        logger.info(f"Retrieval: top-k={retrieval['k']}")

    prompt_template = """Du bist ein spezialisierter CTI-Analyst. Beantworte die Frage basierend auf den gegebenen Kontextinformationen.

        STRENGE REGELN:
        1. Beantworte die Frage AUSSCHLIESSLICH mit Informationen aus dem bereitgestellten Kontext
        2. Wenn der Kontext keine relevanten Informationen enthält, antworte: "Ich habe dazu keine Informationen in den CTI-Berichten gefunden."
        3. Erfinde KEINE Informationen, extrapoliere nicht und spekuliere nicht
        4. Bleibe faktenbasiert und präzise
        5. Verwende die gleiche Sprache wie die Frage (Deutsch/Englisch)

        KONTEXT:
        {context}

        FRAGE: {question}

        CRITICAL: Wenn keine passenden Informationen im Kontext vorhanden sind, antworte mit "Ich habe dazu keine Informationen in den CTI-Berichten gefunden."

        ANTWORT:"""

    # Chain Type aus Config - Default:stuff
    chain_type = config["llm"].get("chain_type", "stuff")
    logger.info(f"Verwende Chain Type: {chain_type}")

    # Chain erstellen - flexibel für alle Types
    try:
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Für alle Chain Types das gleiche Prompt verwenden
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    except Exception as e:
        logger.error(f"Chain Erstellung fehlgeschlagen: {e}")
        
        # Fallback: Ohne custom prompt versuchen
        logger.info("Versuche Fallback ohne custom prompt...")
        try:
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True
            )
        except Exception as fallback_error:
            logger.error(f"Fallback ebenfalls fehlgeschlagen: {fallback_error}")
            return None
    

# Einmal gebaute RAG-Chain für den Prozess (Index + Embeddings + LLM werden
# sonst bei jeder Frage neu geladen).
_CHAIN_CACHE = {}


def _get_chain():
    """Baut die RAG-Chain einmalig und gibt sie zwischengespeichert zurück."""
    if "chain" in _CHAIN_CACHE:
        return _CHAIN_CACHE["chain"]
    cfg = load_config()
    embeddings = get_embeddings(cfg)
    if embeddings is None:
        return None
    vectorstore = get_vectorstore(cfg, embeddings)
    if vectorstore is None:
        return None
    qa_chain = create_rag_chain(cfg, vectorstore)
    if qa_chain is not None:
        _CHAIN_CACHE["chain"] = qa_chain
    return qa_chain


def answer(query: str, include_sources: bool = False):
    """Frage beantworten """
    if not query or not query.strip():
        return _format_response("Bitte stelle eine Frage.", [], include_sources)

    try:
        qa_chain = _get_chain()
        if qa_chain is None:
            return _format_response("System nicht verfügbar.", [], include_sources)

        # Query verarbeiten
        result = qa_chain.invoke({"query": query})
        answer_text = result["result"].strip()
        sources = result.get("source_documents", [])

        # Antwort validieren: ohne abgerufene Quellen gibt es keine belastbare
        # Antwort (sprachunabhängig, statt auf einen festen Satz zu prüfen).
        if not answer_text or not sources:
            return _format_response("Ich habe dazu keine Informationen gefunden.", [], include_sources)

        return _format_response(answer_text, sources, include_sources)

    except Exception as e:
        logger.error(f"Fehler: {e}")
        
        # Wichtige Fehler erkennen
        error_msg = str(e)
        if any(x in error_msg for x in ["429", "quota", "ResourceExhausted"]):
            return _format_response("System ausgelastet. Bitte später versuchen.", [], include_sources)
        
        return _format_response("Technischer Fehler. Bitte erneut versuchen.", [], include_sources)


def _format_response(answer_text: str, sources: list, include_sources: bool):
    """Strukturierte Response-Erstellung."""
    if not include_sources:
        return answer_text
    
    # Quellen strukturiert aufbereiten
    source_list = []
    seen_files = set()
    
    for doc in sources:
        file_name = doc.metadata.get("source_file", "Unbekannt")
        if file_name not in seen_files:
            source_list.append({
                "file": file_name,
                "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            })
            seen_files.add(file_name)
    
    return {
        "answer": answer_text,
        "sources": source_list,
        "source_count": len(source_list)  
    }

if __name__ == "__main__":
    print("\nCTI-RAG Pipeline")
    print("=" * 50)
    print("Interaktiver RAG-Modus gestartet")
    print("-" * 50)

    try:
        while True:
            q = input("\nFrage (oder 'exit' zum Beenden): ").strip()
            if not q or q.lower() == 'exit':
                break

            print("Suche...", end='', flush=True)
            res = answer(q, include_sources=True)
            print("\r", end='')  # Lösche "Suche... nachdem antwort gezeigt wird"

            # ANTWERT-VERARBEITUNG
            if isinstance(res, dict) and 'answer' in res:
                print(f"Antwort: {res['answer']}")
                # Quellen anzeigen - mit Fehlertoleranz
                sources = res.get('sources', [])
                # Verwende source_count falls vorhanden, sonst zähle die Quellen
                source_count = res.get('source_count', len(sources))
                
                if sources:
                    print(f"\nQuellen ({source_count} Dateien):")
                    seen_files = set()
                    for source in sources:
                        file_name = source.get('file', 'Unbekannt')
                        if file_name not in seen_files:
                            print(f"  - {file_name}")
                            seen_files.add(file_name)
                else:
                    print("\nKeine Quellen verfügbar")
                    
            else:
                # Fallback für String-Antworten oder unerwartete Formate
                print(f"Antwort: {res}")

        print("\nBye!")

    except KeyboardInterrupt:
        print("\n\nUnterbrochen. Bye!")
        
    except Exception as e:
        logger.error(f"Fehler im Hauptprogramm: {e}")
        print(f"\nUnerwarteter Fehler: {e}")
