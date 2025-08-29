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
    def __init__(self, incoherent_attempts: int = 1, always_incoherent: bool = False):
        self._call_counts = {}
        self.incoherent_attempts = incoherent_attempts
        self.always_incoherent = always_incoherent
        logging.info(f"MockCompletions configured with incoherent_attempts={self.incoherent_attempts}, always_incoherent={self.always_incoherent}")

    async def create_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> MockCompletionResponse:
        await asyncio.sleep(0.01)

        if self.always_incoherent:
            return MockCompletionResponse(choices=[MockChoice(message={"content": "Isso também é incoerente..."})])

        user_content = messages[-1]["content"]
        question_key = next((m["content"] for m in messages if m["role"] == "user"), None)
        count = self._call_counts.get(question_key, 0) + 1
        self._call_counts[question_key] = count
        logging.info(f"MockCompletions: Received call #{count} for question: '{question_key}'")

        if "critique" in user_content.lower():
            response_content = "A resposta anterior era vaga e terminava abruptamente."
        elif "base na sua crítica" in user_content.lower():
            response_content = f"Claro, aqui está uma resposta coerente e completa para a sua pergunta: '{question_key}'"
        elif count <= self.incoherent_attempts:
            response_content = "Aprendizado supervisionado é... hmm... e o outro é... não sei..."
        else:
            response_content = f"Esta é uma resposta coerente para '{question_key}'."

        return MockCompletionResponse(choices=[MockChoice(message={"content": response_content})])

class MockChat:
    def __init__(self, incoherent_attempts: int = 1, always_incoherent: bool = False):
        self.completions = MockCompletions(incoherent_attempts=incoherent_attempts, always_incoherent=always_incoherent)

class MockZAIClient:
    def __init__(self, api_key: str, incoherent_attempts: int = 1, always_incoherent: bool = False, **kwargs: Any):
        logging.info(f"MockZAIClient initialized with api_key='{api_key[:4]}...', incoherent_attempts={incoherent_attempts}, always_incoherent={always_incoherent}")
        self.chat = MockChat(incoherent_attempts=incoherent_attempts, always_incoherent=always_incoherent)

class ZAIError(Exception): pass
class RateLimitError(ZAIError): pass
class ServerError(ZAIError): pass

# =================================================
# Coherence Logic Implementation
# =================================================
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_prompts() -> Dict[str, str]:
    """Loads prompts from the YAML configuration file with robust fallbacks."""
    default_prompts = {
        "system_prompt": "Você é um assistente de IA que prioriza a coerência lógica.",
        "reprompt": "Sua resposta anterior está incompleta ou contraditória. Por favor, reformule mantendo coerência total.",
        "critique_prompt": "A resposta anterior foi considerada insatisfatória. Critique-a, identificando as principais falhas de lógica ou coerência.",
        "improve_prompt": "Com base na sua crítica, forneça uma nova resposta que seja completa, coerente e correta."
    }

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        prompts = config.get("prompts", {})

        # Merge loaded prompts with defaults, giving precedence to loaded values
        final_prompts = {**default_prompts, **prompts}

        if not all(k in final_prompts for k in default_prompts):
             logging.warning(
                f"One or more prompts are missing in the configuration file: {CONFIG_PATH}. "
                "Using default prompts for missing keys."
            )

        return final_prompts

    except FileNotFoundError:
        logging.error(f"Configuration file not found at {CONFIG_PATH}. Using default prompts.")
        return default_prompts
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file at {CONFIG_PATH}: {e}. Using default prompts.")
        return default_prompts

def is_coherent(text: str) -> bool:
    if not text:
        return False
    if text.strip().endswith("..."):
        return False
    return True

async def ask_zai(question: str, client: MockZAIClient, max_retries: int = 1) -> str:
    if not question or not question.strip():
        logging.warning("Question is empty or contains only whitespace.")
        return ""

    prompts = load_prompts()
    system_prompt = {"role": "system", "content": prompts["system_prompt"]}
    critique_prompt = prompts["critique_prompt"]
    improve_prompt = prompts["improve_prompt"]

    messages = [system_prompt, {"role": "user", "content": question}]
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

        if attempt < max_retries:
            logging.warning(f"Incoherent response detected. Starting self-correction attempt #{attempt + 1}.")

            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": critique_prompt})

            critique_resp = await client.chat.completions.create_async(
                model="zai-llama3.1-8b", messages=messages.copy()
            )
            critique = critique_resp.choices[0].message["content"]
            logging.info(f"Self-critique received: '{critique}'")

            messages.append({"role": "assistant", "content": critique})
            messages.append({"role": "user", "content": improve_prompt})

            final_resp = await client.chat.completions.create_async(
                model="zai-llama3.1-8b", messages=messages.copy()
            )
            answer = final_resp.choices[0].message["content"]
            logging.info(f"Improved answer received: '{answer}'")

            if is_coherent(answer):
                is_final_answer_coherent = True
                break

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
