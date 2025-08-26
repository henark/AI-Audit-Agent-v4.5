import sys
import os
import pytest

# Add the project's 'src' directory to the Python path
# This allows us to import 'coherence_example' from the tests directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import coherence_example

# Mark the whole module as async, which is a convention for pytest-asyncio
pytestmark = pytest.mark.asyncio

async def test_ask_zai_handles_incoherent_response(mocker):
    """
    Given an incoherent first response from the client,
    ask_zai should re-prompt to get a coherent answer.
    """
    # ARRANGE
    # Spy on the method that simulates the external API call to track its usage.
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')

    # The client is configured to be incoherent on the first call by default.
    client = coherence_example.MockZAIClient(api_key="test_key_incoherent")
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # ACT
    final_answer = await coherence_example.ask_zai(question, client)

    # ASSERT
    # 1. Check that the API was called twice (initial prompt + re-prompt).
    assert spy_create_async.call_count == 2, "Expected two calls to the API for re-prompting."

    # 2. Check that the final answer is the coherent one from the second call.
    assert "Claro, aqui está uma resposta coerente" in final_answer, "The final answer should be the coherent one."

    # 3. Inspect the messages for each call to verify the conversation history.
    first_call_messages = spy_create_async.call_args_list[0].kwargs['messages']
    second_call_messages = spy_create_async.call_args_list[1].kwargs['messages']

    # The first call should have 2 messages: system and user.
    assert len(first_call_messages) == 2
    assert first_call_messages[1]['role'] == 'user'
    assert first_call_messages[1]['content'] == question

    # The second call should have 4 messages, preserving the full history.
    assert len(second_call_messages) == 4, "The second call should contain the full conversation history."
    assert second_call_messages[0]['role'] == 'system'
    assert second_call_messages[1]['role'] == 'user'
    assert second_call_messages[1]['content'] == question, "History should contain original question."
    assert second_call_messages[2]['role'] == 'assistant', "History should contain the bad answer."
    assert "não sei..." in second_call_messages[2]['content']
    assert second_call_messages[3]['role'] == 'user', "History should end with the re-prompt."
    assert "reformule" in second_call_messages[3]['content']


async def test_ask_zai_skips_repompt_for_coherent_response(mocker):
    """
    Given a coherent first response from the client,
    ask_zai should not re-prompt and should return the first answer.
    """
    # ARRANGE
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')

    # Configure the client to be coherent from the start.
    client = coherence_example.MockZAIClient(api_key="test_key_coherent", start_incoherent=False)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # ACT
    final_answer = await coherence_example.ask_zai(question, client)

    # ASSERT
    # 1. Check that the API was called only once.
    assert spy_create_async.call_count == 1, "Expected only one call to the API for a coherent response."

    # 2. Check that the final answer is the coherent one from the first call.
    assert "Esta é uma resposta coerente" in final_answer
    assert question in final_answer
