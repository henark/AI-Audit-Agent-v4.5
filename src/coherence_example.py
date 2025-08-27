import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

# =================================================
# Mock ZAI SDK Implementation (Refactored for Testability)
# =================================================

@dataclass
class MockMessage:
    content: str

@dataclass
class MockChoice:
    message: Dict[str, str]

@dataclass
class MockCompletionResponse:
    choices: List[MockChoice]
    model: str = "mock-zai-model"

class MockCompletions:
    def __init__(self, incoherent_attempts: int = 1):
        self._call_counts = {}
        self.incoherent_attempts = incoherent_attempts
        logging.info(f"MockCompletions configured with incoherent_attempts={self.incoherent_attempts}")

    async def create_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> MockCompletionResponse:
        await asyncio.sleep(0.01)
        user_content = messages[-1]["content"]

        # Find the first user message in the history to use as a stable key.
        question_key = None
        for message in messages:
            if message.get('role') == 'user':
                question_key = message.get('content')
                break

        count = self._call_counts.get(question_key, 0) + 1
        self._call_counts[question_key] = count

        logging.info(f"MockCompletions: Received call #{count} for question: '{question_key}'")

        if count <= self.incoherent_attempts:
            response_content = "Aprendizado supervisionado é... hmm... e o outro é... não sei..."
        else:
            # Once we are past the incoherent attempts, the response depends on the prompt.
            if "reformule" in user_content.lower():
                response_content = f"Claro, aqui está uma resposta coerente e completa para a sua pergunta: '{question_key}'"
            else:
                # This would be the first, and coherent, response.
                response_content = f"Esta é uma resposta coerente para '{question_key}'."

        return MockCompletionResponse(choices=[MockChoice(message={"content": response_content})])

class MockChat:
    def __init__(self, incoherent_attempts: int = 1):
        self.completions = MockCompletions(incoherent_attempts=incoherent_attempts)

class MockZAIClient:
    def __init__(self, api_key: str, incoherent_attempts: int = 1, **kwargs: Any):
        logging.info(f"MockZAIClient initialized with api_key='{api_key[:4]}...', incoherent_attempts={incoherent_attempts}")
        self.chat = MockChat(incoherent_attempts=incoherent_attempts)

class ZAIError(Exception): pass
class RateLimitError(ZAIError): pass
class ServerError(ZAIError): pass

# =================================================
# Coherence Logic Implementation
# =================================================

DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": "Você é um assistente de IA que prioriza a coerência lógica."
}

def is_coherent(text: str) -> bool:
    if not text:
        return False
    if text.strip().endswith("..."):
        return False
    return True

async def ask_zai(question: str, client: MockZAIClient, max_retries: int = 2) -> str:
    if not question or not question.strip():
        logging.warning("Question is empty or contains only whitespace.")
        return ""

    messages = [DEFAULT_SYSTEM_PROMPT, {"role": "user", "content": question}]

    answer = ""
    is_final_answer_coherent = False

    for attempt in range(max_retries + 1):
        logging.info(f"--- Asking question (Attempt #{attempt + 1}): '{question}' ---")

        resp = await client.chat.completions.create_async(
            model="zai-llama3.1-8b", messages=messages.copy()
        )
        answer = resp.choices[0].message["content"]
        logging.info(f"Response received (Attempt #{attempt + 1}): '{answer}'")

        if is_coherent(answer):
            logging.info("Coherent response received.")
            is_final_answer_coherent = True
            break
        else:
            logging.warning(f"Incoherent response detected on attempt #{attempt + 1}.")
            # Append the assistant's bad answer and the user's correction to the message history
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": "Sua resposta anterior está incompleta ou contraditória. Por favor, reformule mantendo coerência total."
            })

    if not is_final_answer_coherent:
        logging.error(f"Failed to get a coherent answer after {max_retries + 1} attempts.")

    return answer.strip()

# =================================================
# Demonstration
# =================================================
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # The demonstration will use the default behavior (1 incoherent response)
    client = MockZAIClient(api_key="DUMMY_API_KEY", incoherent_attempts=1)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."
    final_answer = await ask_zai(question, client)

    print("\n=========================")
    print("=== Resposta Coerente ===")
    print("=========================\n")
    print(final_answer)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Ensure all log handlers are flushed before the script exits.
        logging.shutdown()
