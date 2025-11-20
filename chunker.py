# chunker.py
import re

"""
Chunker Module
--------------
- 문서를 논리 단위(Section) 기준으로 분리한 뒤
  max_tokens / overlap 기반으로 chunking
- Section-aware chunking (Strategy B)이 기본값
"""

DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP = 100


def split_sections(text: str):
    """
    문서를 section header 기준으로 1차 분리.
    '##', '###', '\n# ' 등 다양한 헤더 패턴을 포착.
    """
    # 다양한 헤더 패턴 지원
    pattern = r"(?:\n#+\s+|\n##|\n###|\n# )"
    sections = re.split(pattern, text)
    sections = [s.strip() for s in sections if s.strip()]
    return sections


def chunk_tokens(tokens, max_tokens, overlap):
    """
    token 리스트를 window 방식으로 chunking.
    """
    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = start + max_tokens
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap

    return chunks


def chunk_document(text: str,
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   overlap: int = DEFAULT_OVERLAP):
    """
    Section-aware chunking (기본 전략)
    1) section 단위로 잘라내고
    2) 각 section을 token window 방식으로 chunk
    """
    all_chunks = []

    sections = split_sections(text)
    if not sections:  # fallback
        sections = [text]

    for sec in sections:
        tokens = sec.split()
        sec_chunks = chunk_tokens(tokens, max_tokens, overlap)
        all_chunks.extend(sec_chunks)

    return all_chunks


# Optional: 다른 chunk 전략을 쉽게 붙일 수 있도록 함수 래핑
def get_chunker(strategy: str = "B"):
    if strategy == "B":  # Section-aware
        return chunk_document
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
