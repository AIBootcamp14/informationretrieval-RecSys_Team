import json
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import kss
except ImportError:  # pragma: no cover
    kss = None


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    if kss is not None:
        try:
            return [s.strip() for s in kss.split_sentences(text) if s.strip()]
        except Exception:
            pass
    # Fallback: simple period-based split
    parts = [p.strip() for p in text.replace("?", ".").replace("!", ".").split(".")]
    return [p for p in parts if p]


def build_sentence_chunks(sentences: Sequence[str], docid: str, title: str = "") -> List[Dict]:
    chunks: List[Dict] = []
    # single sentences
    for idx, sent in enumerate(sentences):
        text = f"{title} {sent}".strip() if title else sent
        chunks.append(
            {
                "docid": docid,
                "chunk_id": f"{docid}::s{idx}",
                "title": title,
                "text": text,
                "type": "sent",
            }
        )
    # sliding window of 2 sentences
    for idx in range(len(sentences) - 1):
        combo = sentences[idx] + " " + sentences[idx + 1]
        text = f"{title} {combo}".strip() if title else combo
        chunks.append(
            {
                "docid": docid,
                "chunk_id": f"{docid}::s{idx}_{idx+1}",
                "title": title,
                "text": text,
                "type": "sent2",
            }
        )
    return chunks


def load_documents(path: Path) -> List[Dict]:
    docs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    for doc in documents:
        docid = doc.get("docid") or doc.get("id") or doc.get("uid")
        if not docid:
            continue
        title = doc.get("title") or ""
        content = doc.get("content") or doc.get("text") or ""
        sentences = split_sentences(content)
        chunks.extend(build_sentence_chunks(sentences, docid, title=title))
    return chunks


__all__ = ["load_documents", "chunk_documents", "split_sentences"]
