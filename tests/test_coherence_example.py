import sys
import os
import pytest

# Add the project's 'src' directory to the Python path
# This allows us to import 'coherence_example' from the tests directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import coherence_example

# Mark the whole module as async, which is a convention for pytest-asyncio
pytestmark = pytest.mark.asyncio

# A dictionary of mock prompts to be returned by the mocked `load_prompts` function.
MOCK_PROMPTS = {
    "system_prompt": "Você é um testador de IA.",
    "reprompt": "Sua resposta anterior está incorreta. Por favor, reformule.",
    "critique_prompt": "Critique a resposta anterior.",
    "improve_prompt": "Com base na sua crítica, melhore a resposta."
}


async def test_ask_zai_uses_self_correction_on_incoherent_response(mocker):
    """
    Given an incoherent first response, ask_zai should use the self-correction
    mechanism (critique and improve) to get a coherent answer.
    """
    # ARRANGE
    mocker.patch('coherence_example.load_prompts', return_value=MOCK_PROMPTS)
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')
    client = coherence_example.MockZAIClient(api_key="test_key_incoherent", incoherent_attempts=1)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # ACT
    final_answer = await coherence_example.ask_zai(question, client, max_retries=1)

    # ASSERT
    # Expect 3 calls: initial attempt (fail), critique (fail), improve (succeed)
    assert spy_create_async.call_count == 3, "Expected three calls for self-correction."
    assert "Claro, aqui está uma resposta coerente" in final_answer

    # Call 1: Initial question
    messages_1 = spy_create_async.call_args_list[0].kwargs['messages']
    assert len(messages_1) == 2
    assert messages_1[0]['content'] == MOCK_PROMPTS["system_prompt"]
    assert messages_1[1]['content'] == question

    # Call 2: Critique prompt
    messages_2 = spy_create_async.call_args_list[1].kwargs['messages']
    assert len(messages_2) == 4
    assert messages_2[2]['role'] == 'assistant' # Bad answer
    assert messages_2[3]['content'] == MOCK_PROMPTS["critique_prompt"]

    # Call 3: Improve prompt
    messages_3 = spy_create_async.call_args_list[2].kwargs['messages']
    assert len(messages_3) == 6
    assert messages_3[4]['role'] == 'assistant' # Critique
    assert messages_3[5]['content'] == MOCK_PROMPTS["improve_prompt"]


async def test_ask_zai_handles_empty_question(mocker):
    """
    Given an empty or whitespace-only question,
    ask_zai should return an empty string without calling the config or the API.
    """
    # ARRANGE
    mock_load_prompts = mocker.patch('coherence_example.load_prompts')
    client = coherence_example.MockZAIClient(api_key="test_key_empty")

    # ACT & ASSERT for empty string
    final_answer_empty = await coherence_example.ask_zai("", client)
    assert final_answer_empty == ""
    mock_load_prompts.assert_not_called()

    # ACT & ASSERT for whitespace-only string
    final_answer_whitespace = await coherence_example.ask_zai("   ", client)
    assert final_answer_whitespace == ""
    mock_load_prompts.assert_not_called()


async def test_ask_zai_skips_repompt_for_coherent_response(mocker):
    """
    Given a coherent first response from the client,
    ask_zai should not re-prompt and should return the first answer.
    """
    # ARRANGE
    mocker.patch('coherence_example.load_prompts', return_value=MOCK_PROMPTS)
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')
    client = coherence_example.MockZAIClient(api_key="test_key_coherent", incoherent_attempts=0)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # ACT
    final_answer = await coherence_example.ask_zai(question, client)

    # ASSERT
    assert spy_create_async.call_count == 1
    assert "Esta é uma resposta coerente" in final_answer
    assert question in final_answer

    first_call_messages = spy_create_async.call_args_list[0].kwargs['messages']
    assert first_call_messages[0]['content'] == MOCK_PROMPTS["system_prompt"]


async def test_ask_zai_fails_after_persistent_incoherence(mocker):
    """
    Given a client that always returns incoherent responses,
    ask_zai should fail gracefully after exhausting its retries.
    """
    # ARRANGE
    mocker.patch('coherence_example.load_prompts', return_value=MOCK_PROMPTS)
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')

    # Configure client to be always incoherent
    client = coherence_example.MockZAIClient(api_key="test_key_always_incoherent", always_incoherent=True)
    question = "This will always fail."

    # ACT
    final_answer = await coherence_example.ask_zai(question, client, max_retries=1)

    # ASSERT
    # With max_retries=1, we expect 4 calls:
    # 1. Initial attempt (fails)
    # 2. Critique call (part of first retry)
    # 3. Improve call (part of first retry)
    # 4. A final attempt in the loop (fails)
    assert spy_create_async.call_count == 4

    # The final answer should be the last incoherent response.
    assert "incoerente..." in final_answer
