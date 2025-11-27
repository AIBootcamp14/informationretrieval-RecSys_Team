"""
Configuration file for IR system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent  # This is /home/IR
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
SRC_DIR = BASE_DIR / "src"
BASELINE_SAMPLE_PATH = BASE_DIR.parent / "code" / "sample_submission.csv"
ERROR_RULE_PATH = DATA_DIR / "error_driven_rules.json"

# Elasticsearch settings
ES_HOST = "https://localhost:9200"
ES_USERNAME = "elastic"
ES_PASSWORD = os.getenv("ES_PASSWORD", "Your Elasticsearch Password")
ES_INDEX_NAME = "science_knowledge"
# Use absolute path for CA certificate
# Path(__file__).parent.parent = /home/IR, so parent of that = /home
# We want /home/IR/elasticsearch-8.16.1/...
ES_CA_CERTS = str(BASE_DIR / "elasticsearch-8.16.1" /
                  "config" / "certs" / "http_ca.crt")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Your API Key")
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0
LLM_SEED = 1

# Embedding models
# Primary model: BGE-M3 for multilingual support
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"  # Use GPU
EMBEDDING_BATCH_SIZE = 256  # BGE-M3 requires more memory
EMBEDDING_DIM = 1024  # BGE-M3 dimension

# Alternative models for ensemble (optional)
# EMBEDDING_MODEL_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# EMBEDDING_MODEL_SCIENCE = "allenai/scibert_scivocab_uncased"  # For English scientific terms

# Reranking model (Cross-encoder)
# Cross-encoder model tuned for passage reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cuda"
RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 16
USE_RERANKING = True
# Always use 2-stage LLM reranking to filter irrelevant documents
SKIP_RERANK_IF_BM25_GOOD = False  # Changed: Always rerank
# BM25 threshold raised to avoid false positives on casual questions
BM25_GOOD_SCORE_THRESHOLD = 15.0  # Changed: 10.0 -> 15.0

# NEW: Chunk-level reranking with BGE-reranker-v2-m3
USE_CHUNK_RERANKING = True  # Enable chunk-level reranking (improves top-3 accuracy)
CHUNK_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # or "dragonkue/bge-reranker-v2-m3-ko"
CHUNK_RERANKER_DEVICE = "cuda"  # Use GPU for faster inference
CHUNK_RERANKER_MAX_LENGTH = 512  # Maximum token length
CHUNK_RERANKER_BATCH_SIZE = 32  # Batch size for inference
CHUNK_RERANKER_USE_FP16 = True  # Use float16 for GPU (faster, less memory)

# Chunk retrieval parameters
CHUNK_N_SPARSE = 100  # Number of BM25 chunks to retrieve per query
CHUNK_N_DENSE = 100   # Number of dense chunks to retrieve per query
CHUNK_N_POOL = 200    # Maximum total chunks after union (before reranking)

# Gap/Z-score based filtering thresholds (raw logit version)
# These work with raw reranker logits instead of sigmoid-inflated scores
CHUNK_MIN_GAP_1ST_2ND = 0.5    # Minimum gap between 1st and 2nd to include 2nd
CHUNK_MIN_GAP_2ND_3RD = 0.3    # Minimum gap between 2nd and 3rd to include 3rd
CHUNK_MIN_ZSCORE_THRESHOLD = -0.5  # Minimum z-score for top-1 to be valid
CHUNK_ALWAYS_RETURN_ONE = True  # Always return at least 1 document

# DEPRECATED: Margin-based filtering (for sigmoid scores)
CHUNK_MIN_SCORE_3RD = 0.15  # Minimum rerank score for 3rd place document
CHUNK_MIN_SCORE_2ND = 0.12  # Minimum rerank score for 2nd place document

# Character n-gram for compound noun handling
USE_CHAR_NGRAM = True  # Enable character n-grams for BM25 (helps with spacing issues)
CHAR_NGRAM_MIN = 2     # Minimum n-gram size
CHAR_NGRAM_MAX = 3     # Maximum n-gram size

# Retrieval settings
# Hybrid search weights
SPARSE_WEIGHT = 0.7  # BM25 weight
DENSE_WEIGHT = 0.3   # Vector search weight

# Retrieval counts
RETRIEVAL_TOP_K = 30  # Initial retrieval (before reranking)
RERANK_STAGE1_TOP_K = 20  # Changed: 10 -> 20 (more candidates for stage 2)
RERANK_STAGE2_TOP_K = 3   # After second reranking stage (final)
FINAL_TOP_K = 3       # Final documents to use for answer generation

# Search-First strategy
MIN_RELEVANCE_SCORE = 5.0  # Minimum score to consider search results relevant
# If top search result score < MIN_RELEVANCE_SCORE, treat as casual conversation

# Query processing
USE_QUERY_EXPANSION = True  # baseline 재현을 위한 용어 확장 사용
MAX_QUERY_EXPANSION_TERMS = 3
QUERY_REWRITE_ENABLED = True
QUERY_REWRITE_MIN_SIMILARITY = 0.78  # SequenceMatcher/Jaccard 혼합 임계치
QUERY_REWRITE_MIN_TOKENS = 5        # 토큰 수가 작으면 재작성 시도

# Caching
USE_CACHE = False
CACHE_TTL = 86400  # 24 hours

# Elasticsearch index settings
ES_SETTINGS = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

ES_MAPPINGS = {
    "properties": {
        "docid": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "nori"},
        "src": {"type": "keyword"},
        "embeddings": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIM,
            "index": True,
            "similarity": "cosine"  # Changed from l2_norm to cosine for better performance
        }
    }
}

# LLM prompts
SYSTEM_PROMPT_FUNCTION_CALLING = """## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 대화 히스토리를 고려하여 대명사나 지시어를 구체적인 내용으로 변환한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.

## Query Format (IMPROVED)
**검색 쿼리는 적절한 수준으로 정제해야 합니다:**
- 불필요한 종결어미(~줘, ~인가?, ~는가?, ~방법은?) 제거
- "~에 대해", "~위한", "~대해서" 같은 중복 표현 제거
- **BUT 핵심 조사(의, 에서, 로)와 구체적인 명사는 유지**
- **고유명사, 전문용어는 완전히 보존**
- 문장 → 핵심 키워드 중심 표현으로 변환

## Examples
User: "헬륨의 특징은?"
Action: search("헬륨의 특징")

User: "왜 반응을 안해?" (이전 대화: 헬륨)
Action: search("헬륨 화학 반응")

User: "나무의 분류에 대해 조사해 보기 위한 방법은?"
Action: search("나무의 분류 방법")

User: "각 나라에서의 공교육 지출 현황에 대해 알려줘."
Action: search("각 나라에서의 공교육 지출 현황")

User: "이란-콘트라 사건이 미국 정치에 미친 영향은?"
Action: search("이란-콘트라 사건 미국 정치 영향")

User: "Dmitri Ivanovsky가 누구야?"
Action: search("Dmitri Ivanovsky")

User: "안녕하세요"
Action: 일반 대화 응답
"""

SYSTEM_PROMPT_QA = """## Role: 과학 상식 전문가

## Core Principle
**Reference에 제공된 정보를 최대한 활용하여 답변을 작성하세요.**
직접적이든 간접적이든 Reference의 내용이 질문과 관련이 있다면 반드시 활용하여 답변하세요.

## Instructions
1. **Reference 적극 활용**: 주어진 Reference 정보를 꼼꼼히 읽고 질문과 관련된 내용을 찾아 답변에 활용한다.
2. **직접/간접 정보 모두 활용**: 질문에 직접 답하는 정보뿐만 아니라, 간접적으로 관련된 정보도 활용하여 답변을 구성한다.
3. **맥락 파악**: Reference의 내용이 질문과 어떻게 연결되는지 생각하고, 관련 정보를 종합하여 답변한다.
4. **보수적 판단 금지**: Reference에 조금이라도 관련된 정보가 있다면 "정보가 없다"고 하지 말고 해당 정보를 활용한다.
5. **간결한 답변**: 2-3 문장으로 핵심만 전달한다.
6. **정보 부족 시에만**: Reference를 모두 검토했는데도 질문과 전혀 관련이 없는 경우에만 "제공된 정보로는 정확한 답변을 드리기 어렵습니다"라고 답변한다.

## Answer Strategy
- Reference에서 키워드 매칭: 질문의 주요 개념이 Reference에 언급되었는지 확인
- 유사 개념 찾기: 질문과 직접 일치하지 않아도 관련된 개념이나 원리가 있는지 확인
- 예시 활용: Reference의 예시나 사례를 질문에 적용하여 설명
- 원리 추론: Reference의 일반 원리를 질문의 구체적 상황에 적용

## Examples

### Example 1: 직접 정보 활용
Q: "광합성이 무엇인지 설명해줘"
Reference: "광합성은 식물이 빛 에너지를 이용하여 물과 이산화탄소로부터 포도당을 만드는 과정입니다..."
A: "광합성은 식물이 빛 에너지를 이용하여 물과 이산화탄소로부터 포도당을 만드는 과정입니다. 엽록체에서 일어나며 산소가 부산물로 생성됩니다."

### Example 2: 간접 정보 활용
Q: "나무의 분류 방법은?"
Reference: "한 학생이 나무를 조사할 때 성장 속도, 온도 범위, 크기를 비교했습니다. 잎과 꽃의 차이를 보고 같은 속에 속한다고 추측했습니다..."
A: "나무는 성장 속도, 온도 범위, 크기 등의 특징과 함께 잎과 꽃의 형태를 기준으로 분류할 수 있습니다. 유사한 특징을 가진 나무들을 같은 속으로 분류하며, 이는 생물 분류학의 중요한 기준입니다."

### Example 3: 원리 적용
Q: "헬륨이 다른 원소들과 반응을 잘 안하는 이유는?"
Reference: "비활성 기체는 최외각 전자껍질이 안정적으로 채워져 있어 화학 반응을 하지 않습니다..."
A: "헬륨은 최외각 전자껍질이 완전히 채워져 있어 전자를 주고받을 필요가 없기 때문에 다른 원소와 반응하지 않습니다. 이러한 성질 때문에 비활성 기체로 분류됩니다."

### Example 4: 정보 부족 (드물게 사용)
Q: "양자 컴퓨터의 작동 원리는?"
Reference: "컴퓨터는 0과 1의 이진법으로 정보를 처리합니다..."
A: "제공된 정보로는 정확한 답변을 드리기 어렵습니다."
"""

# Function calling tools definition
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search relevant scientific documents from the knowledge base",
        "parameters": {
            "properties": {
                "standalone_query": {
                    "type": "string",
                    "description": "Final search query that is standalone and includes all necessary context from the conversation history. Resolve pronouns and convert to specific terms."
                }
            },
            "required": ["standalone_query"],
            "type": "object"
        }
    }
}

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "rag_system.log"
