from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.schema import Document
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_document(file_path: str) -> List[Document]:
    """Charge un seul document selon son extension."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(str(path))
    elif ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Format non supporté : {ext}")

    docs = loader.load()
    logger.info(f"Chargé : {path.name}  ({len(docs)} pages/sections)")
    return docs


def load_directory(dir_path: str) -> List[Document]:
    """Charge tous les documents d'un dossier (PDF, DOCX, TXT)."""
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dir_path}")

    all_docs = []

    for file in path.rglob("*"):
        if file.suffix.lower() in [".pdf", ".docx", ".doc", ".txt"]:
            try:
                docs = load_document(str(file))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Erreur sur {file.name} : {e}")

    logger.info(f"Total documents chargés : {len(all_docs)}")
    return all_docs