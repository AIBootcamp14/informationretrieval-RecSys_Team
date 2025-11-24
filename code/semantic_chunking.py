"""
Semantic Chunking for Scientific Korean Text

핵심 기능:
1. 문장 단위 분리 (한국어 특화)
2. 의미적 유사도 기반 그룹화
3. 과학 개념 완결성 보장
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer

# 전역 임베딩 모델 (한 번만 로드)
_embedding_model = None

def get_embedding_model():
    """임베딩 모델 싱글톤"""
    global _embedding_model
    if _embedding_model is None:
        print("📥 한국어 임베딩 모델 로드 중... (jhgan/ko-sroberta-multitask)")
        _embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        print("✅ 모델 로드 완료")
    return _embedding_model

def split_sentences_korean(text):
    """
    한국어 문장 분리 (규칙 기반)

    KoNLPy는 느리고 과학 용어 처리 미흡하므로
    규칙 기반으로 빠르고 정확하게 처리
    """
    if not text or len(text.strip()) == 0:
        return []

    # 문장 종결 패턴
    # 1. 마침표, 느낌표, 물음표 뒤 공백 or 줄바꿈
    # 2. 단, 소수점(3.14), 약어(Dr.), 순서(1.) 제외

    # 임시 보호: 숫자.숫자 → 숫자<DOT>숫자
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', text)

    # 문장 분리
    sentences = re.split(r'([.!?])\s+', text)

    # 문장 종결 부호 다시 붙이기
    result = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue

        # 다음이 구두점이면 붙이기
        if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
            sent = sent + sentences[i + 1]
            i += 2
        else:
            i += 1

        # <DOT> 복원
        sent = sent.replace('<DOT>', '.')

        if len(sent.strip()) > 0:
            result.append(sent.strip())

    return result

def cosine_similarity(emb1, emb2):
    """코사인 유사도 계산"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def semantic_sentence_chunking(document, max_chunk_size=400, min_chunk_size=100, similarity_threshold=0.65, overlap_sentences=0):
    """
    의미적 문장 청킹 with Overlap 지원

    Args:
        document: 원본 텍스트
        max_chunk_size: 최대 청크 크기 (문자 수)
        min_chunk_size: 최소 청크 크기 (너무 작은 청크 방지)
        similarity_threshold: 의미적 유사도 임계값 (낮을수록 개념 전환 민감)
        overlap_sentences: 청크 간 중복할 문장 수 (0=중복 없음, 1=마지막 1문장 중복 등)

    Returns:
        List[str]: 의미적으로 완결된 청크들
    """
    if not document or len(document.strip()) == 0:
        return []

    # 짧은 문서는 그대로 반환
    if len(document) <= max_chunk_size:
        return [document]

    # Step 1: 문장 분리
    sentences = split_sentences_korean(document)

    if len(sentences) == 0:
        return [document]

    if len(sentences) == 1:
        # 문장이 1개면 그대로 반환
        return [sentences[0]]

    # Step 2: 각 문장 임베딩
    model = get_embedding_model()
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Step 3: 의미적 유사도 기반 그룹화 (with Overlap)
    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])
    overlap_buffer = []  # 다음 청크에 포함할 중복 문장들

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_len = len(sent)

        # 현재 청크에 추가 시 크기 체크
        if current_length + sent_len <= max_chunk_size:
            # 크기 OK → 의미적 유사도 체크
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            similarity = cosine_similarity(prev_emb, curr_emb)

            if similarity >= similarity_threshold:
                # 유사도 높음 → 같은 개념 → 추가
                current_chunk.append(sent)
                current_length += sent_len
            else:
                # 유사도 낮음 → 개념 전환 → 새 청크
                # 단, 현재 청크가 너무 작으면 강제 추가
                if current_length < min_chunk_size:
                    current_chunk.append(sent)
                    current_length += sent_len
                else:
                    # 청크 저장
                    chunks.append(' '.join(current_chunk))

                    # Overlap 처리: 마지막 N개 문장을 다음 청크 시작으로
                    if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                        overlap_buffer = current_chunk[-overlap_sentences:]
                        current_chunk = overlap_buffer + [sent]
                        current_length = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk = [sent]
                        current_length = sent_len
        else:
            # 크기 초과 → 무조건 새 청크
            chunks.append(' '.join(current_chunk))

            # Overlap 처리
            if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                overlap_buffer = current_chunk[-overlap_sentences:]
                current_chunk = overlap_buffer + [sent]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk = [sent]
                current_length = sent_len

    # 마지막 청크
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def analyze_chunking(original_doc, chunks):
    """청킹 결과 분석 (디버깅용)"""
    print(f"\n{'='*80}")
    print(f"청킹 분석")
    print(f"{'='*80}")
    print(f"원본 길이: {len(original_doc)}자")
    print(f"청크 수: {len(chunks)}개")
    print(f"\n각 청크:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[청크 {i}] ({len(chunk)}자)")
        print(f"{chunk[:100]}...")  # 첫 100자만 출력

def test_semantic_chunking():
    """테스트 함수"""
    print("="*80)
    print("Semantic Chunking 테스트")
    print("="*80)

    # 테스트 문서 (DNA + 광합성 혼재)
    test_doc = """DNA는 유전정보를 저장하는 이중나선 구조의 분자이다.
DNA 복제는 반보존적 방식으로 진행되며, 각 가닥이 주형 역할을 한다.
DNA 중합효소는 새로운 가닥을 합성하는 핵심 효소이다.
식물의 광합성은 빛 에너지를 화학 에너지로 변환하는 과정이다.
광합성은 엽록소에서 일어나며 명반응과 암반응으로 나뉜다.
명반응에서는 물이 분해되어 산소가 발생하고 ATP가 생성된다."""

    print(f"\n📄 원본 문서 ({len(test_doc)}자):")
    print(test_doc)

    # 청킹 실행 (overlap 없음)
    print(f"\n{'='*80}")
    print("테스트 1: Overlap 없음")
    print(f"{'='*80}")
    chunks_no_overlap = semantic_sentence_chunking(test_doc, max_chunk_size=200, overlap_sentences=0)
    analyze_chunking(test_doc, chunks_no_overlap)

    # 청킹 실행 (overlap 1문장)
    print(f"\n{'='*80}")
    print("테스트 2: Overlap 1문장")
    print(f"{'='*80}")
    chunks_overlap1 = semantic_sentence_chunking(test_doc, max_chunk_size=200, overlap_sentences=1)
    analyze_chunking(test_doc, chunks_overlap1)

    print(f"\n{'='*80}")
    print("✅ 테스트 완료")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_semantic_chunking()
