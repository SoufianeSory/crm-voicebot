from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Découpe les documents en chunks avec overlap.
    RecursiveCharacterTextSplitter essaie de couper sur les
    paragraphes, phrases, puis mots — dans cet ordre.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Nettoyage : supprimer les chunks trop courts (bruit)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

    logger.info(f"Chunks générés : {len(chunks)}  "
                f"(taille moy. : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars)")
    return chunks