"""
src/ingestion/splitter.py — v2 avec détection FAQ intelligente.

Problème résolu : RecursiveCharacterTextSplitter mélange plusieurs Q/R
dans un même chunk → le bon chunk est noyé dans le bruit sémantique.

Solution : 1 chunk par paire Q/R pour les documents FAQ.
Fallback RecursiveCharacterTextSplitter pour les autres documents.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import re
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def _is_faq_format(text: str) -> bool:
    """Détecte si le document est au format Q:/R: (FAQ structurée)."""
    lines = text.split('\n')[:30]
    for line in lines:
        if re.match(r'^\s*Q\s*[\:\.]', line.strip(), re.IGNORECASE):
            return True
    return False


def _split_faq(text: str, metadata: dict) -> List[Document]:
    """
    Découpe une FAQ en chunks 1-par-paire-Q/R.
    Chaque chunk = 1 question + sa réponse = signal sémantique pur.
    """
    # Normaliser les variantes : "Q :" → "Q:"  |  "R :" → "R:"
    text = re.sub(r'\bQ\s*:', 'Q:', text)
    text = re.sub(r'\bR\s*:', 'R:', text)

    # Séparer à chaque début de Q:
    raw_blocks = re.split(r'(?=\nQ:|\AQ:)', text)

    chunks = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if not block.startswith('Q:'):
            if len(block) > 60:
                chunks.append(Document(page_content=block, metadata=metadata))
            continue
        if len(block) > 30:
            chunks.append(Document(page_content=block, metadata=metadata))

    logger.info(f"[Splitter/FAQ] {len(chunks)} chunks Q/R individuels")
    return chunks


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Découpe intelligente :
    - Documents FAQ (format Q:/R:) → 1 chunk par paire
    - Autres documents             → RecursiveCharacterTextSplitter
    """
    faq_chunks : List[Document] = []
    other_docs : List[Document] = []

    for doc in docs:
        if _is_faq_format(doc.page_content):
            logger.info(f"[Splitter] Format FAQ détecté : {doc.metadata.get('source','?')}")
            faq_chunks.extend(_split_faq(doc.page_content, doc.metadata))
        else:
            other_docs.append(doc)

    other_chunks: List[Document] = []
    if other_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        other_chunks = splitter.split_documents(other_docs)
        other_chunks = [c for c in other_chunks if len(c.page_content.strip()) > 40]

    all_chunks = faq_chunks + other_chunks

    # Déduplification
    seen   : set = set()
    unique : List[Document] = []
    for c in all_chunks:
        key = c.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if unique:
        avg = sum(len(c.page_content) for c in unique) // len(unique)
        logger.info(
            f"[Splitter] {len(unique)} chunks total | "
            f"FAQ : {len(faq_chunks)} | Autres : {len(other_chunks)} | "
            f"Taille moy. : {avg} chars"
        )
    return unique
