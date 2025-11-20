# llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

"""
LLM Module (OpenAI v1.x API)
----------------------------
- standalone query rewriting
- context-based answering
"""

# Load key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_LLM_MODEL = "gpt-4o-mini"


#################################################################
# 1) Standalone Query 생성
#################################################################

def rewrite_query(messages, model=DEFAULT_LLM_MODEL):
    """
    message 배열을 하나의 standalone question으로 변환
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Rewrite the following user messages "
                "into a single standalone question. Do not add or remove meaning."
            )
        },
        {
            "role": "user",
            "content": str(messages)
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.0
    )

    return resp.choices[0].message.content.strip()


#################################################################
# 2) Context-based QA
#################################################################

def answer_with_context(query, context_chunks, model=DEFAULT_LLM_MODEL):
    """
    RAG answer 생성
    """
    context_text = "\n\n".join(
        [f"[Chunk {i}]\n{ck}" for i, ck in enumerate(context_chunks)]
    )

    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert scientific QA assistant. Use ONLY the provided context. "
                "If the answer cannot be found in the context, say '정보를 찾을 수 없습니다.'. "
                "Avoid hallucinations completely."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Context:\n{context_text}\n\n"
                "Answer strictly using the above context."
            )
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.0
    )

    return resp.choices[0].message.content.strip()
