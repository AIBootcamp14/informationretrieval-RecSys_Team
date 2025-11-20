# eval_runner.py
import json
from tqdm import tqdm
from embedder import Embedder
from chunker import chunk_document, get_chunker
from vector_store import VectorStore
from llm import rewrite_query, answer_with_context


"""
Eval Runner
-----------
- 전체 RAG 파이프라인을 orchestration 하는 메인 모듈
- count 옵션으로 스모크 테스트 가능
- 최종 submission.jsonl 생성
"""


############################################################
# 1. 문서 로드
############################################################

def load_documents(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            # 구조: {"docid": "...", "content": "..."}
            docs.append(j)
    return docs


############################################################
# 2. eval.jsonl 로드
############################################################

def load_eval_queries(eval_path):
    queries = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            queries.append(j)
    return queries


############################################################
# 3. RAG Pipeline Runner
############################################################

def eval_rag(
        docs_path,
        eval_path,
        output_path="submission.jsonl",
        embed_model="BAAI/bge-m3",
        chunk_strategy="B",
        topk=5,
        count=5  # 스모크 테스트: N개만 실행. 전체 돌릴 땐 count=None
):

    print("=== Loading Documents ===")
    documents = load_documents(docs_path)

    print("=== Initializing Embedder ===")
    embedder = Embedder(embed_model)

    print(f"=== Initializing Chunker: strategy {chunk_strategy} ===")
    chunk_fn = get_chunker(chunk_strategy)

    print("=== Building Vector Store ===")
    vs = VectorStore(embedder)
    vs.build(documents, chunk_fn=chunk_fn)

    print("=== Loading Evaluation Queries ===")
    queries = load_eval_queries(eval_path)

    print("=== Running Evaluation (RAG) ===")
    with open(output_path, "w", encoding="utf-8") as of:

        for i, item in enumerate(tqdm(queries, desc="RAG Evaluating")):

            # count 제한 (스모크 테스트)
            if (count is not None) and (i >= count):
                print(f"[STOP] Count={count} reached. Ending early.")
                break

            # baseline 형태 유지: msg 배열을 하나의 standalone 문장으로
            messages = item["msg"]
            raw_query = " ".join(m["content"] for m in messages)

            # 1) standalone query rewriting
            standalone = rewrite_query(messages)

            # 2) top-k retrieval
            topk_idx = vs.retrieve_topk(standalone, k=topk)
            topk_docids = [vs.get_doc_id(idx) for idx in topk_idx]

            retrieved_chunks = [vs.get_chunk(idx) for idx in topk_idx]

            # 3) LLM answer generation
            answer = answer_with_context(standalone, retrieved_chunks)

            # 4) output 포맷
            output = {
                "eval_id": item["eval_id"],
                "standalone_query": standalone,
                "topk": topk_docids,
                "answer": answer,
                "references": retrieved_chunks
            }

            of.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"\n=== Done. Output saved to: {output_path} ===")


############################################################
# 4. Main Entry
############################################################

if __name__ == "__main__":

    # 실전 스모크 테스트 (LLM 비용 최소화)
    # eval_rag(
    #     docs_path="data/documents.jsonl",
    #     eval_path="data/eval.jsonl",
    #     output_path="submission_test.csv",
    #     embed_model="BAAI/bge-m3",
    #     chunk_strategy="B",
    #     topk=5,
    #     count=5       # 먼저 5개만 테스트
    # )

    # count=None 으로 전체 실행 시:
    eval_rag(
        docs_path="data/documents.jsonl",
        eval_path="data/eval.jsonl",
        output_path="submission_full.csv",
        embed_model="BAAI/bge-m3",
        chunk_strategy="B",
        topk=5,
        count=None
    )
