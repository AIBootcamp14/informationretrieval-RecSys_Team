"""
LLM client module for function calling and answer generation
"""
from openai import OpenAI
import json
import traceback
from typing import List, Dict, Any, Optional
from loguru import logger
import config.config as cfg


class LLMClient:
    """OpenAI LLM client with enhanced prompts"""

    def __init__(
        self,
        api_key: str = cfg.OPENAI_API_KEY,
        model: str = cfg.LLM_MODEL
    ):
        """
        Initialize LLM client

        Args:
            api_key: OpenAI API key
            model: Model name
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized LLM client with model: {model}")

    def function_calling(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: float = cfg.LLM_TEMPERATURE,
        timeout: int = 60,  # Increased from 10 to 60 seconds
        max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM with function calling capability

        Args:
            messages: Conversation messages
            tools: Available tools
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on timeout

        Returns:
            Response dict or None on error
        """
        if tools is None:
            tools = [cfg.SEARCH_TOOL]

        # Add system prompt
        full_messages = [
            {"role": "system", "content": cfg.SYSTEM_PROMPT_FUNCTION_CALLING}
        ] + messages

        try:
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=full_messages,
                        tools=tools,
                        temperature=temperature,
                        seed=cfg.LLM_SEED,
                        timeout=timeout
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries and "timeout" in str(e).lower():
                        logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                        continue
                    raise  # Re-raise if not timeout or no retries left

            message = response.choices[0].message

            # Check if function was called
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.debug(f"Function called: {function_name} with args: {function_args}")

                return {
                    'type': 'function_call',
                    'function_name': function_name,
                    'arguments': function_args,
                    'content': None
                }
            else:
                # Direct answer without search
                logger.debug("No function call, direct answer")
                return {
                    'type': 'direct_answer',
                    'function_name': None,
                    'arguments': None,
                    'content': message.content
                }

        except Exception as e:
            logger.error(f"LLM function calling error: {e}")
            traceback.print_exc()
            return None

    def generate_answer(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        temperature: float = cfg.LLM_TEMPERATURE,
        timeout: int = 60,  # Increased from 30 to 60 seconds
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Generate answer based on retrieved documents

        Args:
            query: User query
            retrieved_documents: Retrieved and reranked documents
            conversation_history: Previous conversation
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on timeout

        Returns:
            Generated answer or None on error
        """
        # Format retrieved documents
        references = []
        for i, doc in enumerate(retrieved_documents, 1):
            ref = f"[Reference {i}]\n{doc['content']}\n"
            references.append(ref)

        reference_text = "\n".join(references)

        # Build messages
        messages = conversation_history or []

        # Add references as assistant message (simulating retrieval result)
        messages_with_refs = messages + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"검색 결과:\n\n{reference_text}"}
        ]

        # Add system prompt
        full_messages = [
            {"role": "system", "content": cfg.SYSTEM_PROMPT_QA}
        ] + messages_with_refs

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=temperature,
                    seed=cfg.LLM_SEED,
                    timeout=timeout
                )

                answer = response.choices[0].message.content
                logger.debug(f"Generated answer: {answer[:100]}...")

                return answer

            except Exception as e:
                if attempt < max_retries and "timeout" in str(e).lower():
                    logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                    continue
                logger.error(f"LLM answer generation error: {e}")
                traceback.print_exc()
                return None

    def assess_answer_quality(
        self,
        query: str,
        answer: str,
        references: List[str]
    ) -> Dict[str, Any]:
        """
        Assess the quality and relevance of generated answer
        (Optional enhancement for validation)

        Args:
            query: Original query
            answer: Generated answer
            references: Reference documents

        Returns:
            Quality assessment dict
        """
        assessment_prompt = f"""다음 답변의 품질을 평가하세요:

질문: {query}

답변: {answer}

참고 문서: {' '.join(references[:200])}

평가 기준:
1. 관련성: 질문에 대한 답변의 관련성 (1-5)
2. 정확성: 참고 문서 기반 답변의 정확성 (1-5)
3. 완전성: 답변의 완전성 (1-5)

JSON 형식으로 답변:
{{"relevance": 점수, "accuracy": 점수, "completeness": 점수, "confidence": "high/medium/low"}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": assessment_prompt}],
                temperature=0,
                timeout=10
            )

            assessment = json.loads(response.choices[0].message.content)
            return assessment

        except:
            # Default assessment if fails
            return {
                "relevance": 3,
                "accuracy": 3,
                "completeness": 3,
                "confidence": "medium"
            }


# Singleton instance
_llm_client_instance = None

def get_llm_client() -> LLMClient:
    """Get singleton LLM client instance"""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
