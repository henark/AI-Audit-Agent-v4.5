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
    def __init__(self, start_incoherent: bool = True):
        self._call_counts = {}
        self.start_incoherent = start_incoherent
        logging.info(f"MockCompletions configured with start_incoherent={self.start_incoherent}")

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

        if "reformule" in user_content.lower():
            response_content = f"Claro, aqui está uma resposta coerente e completa para a sua pergunta: '{question_key}'"
        elif self.start_incoherent and count == 1:
            response_content = "Aprendizado supervisionado é... hmm... e o outro é... não sei..."
        else:
            response_content = f"Esta é uma resposta coerente para '{question_key}'."

        return MockCompletionResponse(choices=[MockChoice(message={"content": response_content})])

class MockChat:
    def __init__(self, start_incoherent: bool = True):
        self.completions = MockCompletions(start_incoherent=start_incoherent)

class MockZAIClient:
    def __init__(self, api_key: str, start_incoherent: bool = True, **kwargs: Any):
        logging.info(f"MockZAIClient initialized with api_key='{api_key[:4]}...', start_incoherent={start_incoherent}")
        self.chat = MockChat(start_incoherent=start_incoherent)

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

async def ask_zai(question: str, client: MockZAIClient) -> str:
    # An empty or whitespace-only question should not be sent to the API.
    if not question or not question.strip():
        logging.warning("Question is empty or contains only whitespace.")
        return ""

    messages = [DEFAULT_SYSTEM_PROMPT, {"role": "user", "content": question}]

    logging.info(f"--- Asking initial question: '{question}' ---")
    resp = await client.chat.completions.create_async(model="zai-llama3.1-8b", messages=messages.copy())
    answer = resp.choices[0].message["content"]
    logging.info(f"Initial response received: '{answer}'")

    if not is_coherent(answer):
        logging.warning("Incoherent response detected – re-prompting for a better answer.")

        # Append the assistant's bad answer and the user's correction to the message history
        messages.append({"role": "assistant", "content": answer})
        messages.append({
            "role": "user",
            "content": "Sua resposta anterior está incompleta ou contraditória. Por favor, reformule mantendo coerência total."
        })

        resp = await client.chat.completions.create_async(model="zai-llama3.1-8b", messages=messages.copy())
        answer = resp.choices[0].message["content"]
        logging.info(f"Corrected response received: '{answer}'")
    else:
        logging.info("Initial response was coherent.")

    return answer.strip()

# =================================================
# Demonstration
# =================================================
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # The demonstration will use the default behavior (incoherent first)
    client = MockZAIClient(api_key="DUMMY_API_KEY")
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."
    final_answer = await ask_zai(question, client)

    await asyncio.sleep(0.05)

    print("\n=========================")
    print("=== Resposta Coerente ===")
    print("=========================\n")
    print(final_answer)

if __name__ == "__main__":
    asyncio.run(main())
