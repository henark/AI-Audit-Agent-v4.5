import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

# =================================================
# Mock ZAI SDK Implementation
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
    def __init__(self):
        self._call_counts = {}

    async def create_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> MockCompletionResponse:
        await asyncio.sleep(0.01)
        user_content = messages[-1]["content"]

        # FIX: More robustly identify the original question for state tracking.
        match = re.search(r"pergunta '(.*?)'", user_content)
        if match:
            question_key = match.group(1)
        else:
            question_key = user_content

        count = self._call_counts.get(question_key, 0) + 1
        self._call_counts[question_key] = count

        logging.info(f"MockZAIClient: Received call #{count} for question: '{question_key}'")

        if "reformule" in user_content.lower():
            response_content = f"Claro, aqui está uma resposta coerente e completa para a sua pergunta: '{question_key}'"
        elif count == 1:
            response_content = "Aprendizado supervisionado é... hmm... e o outro é... não sei..."
        else:
            response_content = f"Esta é uma resposta coerente para '{question_key}'."

        return MockCompletionResponse(choices=[MockChoice(message={"content": response_content})])

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockZAIClient:
    def __init__(self, api_key: str, **kwargs: Any):
        logging.info(f"MockZAIClient initialized with api_key='{api_key[:4]}...'")
        self.chat = MockChat()

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
    messages = [DEFAULT_SYSTEM_PROMPT, {"role": "user", "content": question}]

    logging.info(f"--- Asking initial question: '{question}' ---")
    resp = await client.chat.completions.create_async(model="zai-llama3.1-8b", messages=messages)
    answer = resp.choices[0].message["content"]
    logging.info(f"Initial response received: '{answer}'")

    if not is_coherent(answer):
        logging.warning("Incoherent response detected – re-prompting for a better answer.")

        # FIX: Include original question in the re-prompt for better context.
        correction_prompt = (
            f"Sua resposta anterior para a pergunta '{question}' foi: '{answer}'. "
            "Esta resposta está incompleta ou contraditória. "
            "Por favor, reformule mantendo coerência total."
        )
        correction_msg = [DEFAULT_SYSTEM_PROMPT, {"role": "user", "content": correction_prompt}]

        resp = await client.chat.completions.create_async(model="zai-llama3.1-8b", messages=correction_msg)
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
    client = MockZAIClient(api_key="DUMMY_API_KEY")
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."
    final_answer = await ask_zai(question, client)

    # FIX: Give logs a moment to flush before printing.
    await asyncio.sleep(0.05)

    print("\n=========================")
    print("=== Resposta Coerente ===")
    print("=========================\n")
    print(final_answer)

if __name__ == "__main__":
    asyncio.run(main())
