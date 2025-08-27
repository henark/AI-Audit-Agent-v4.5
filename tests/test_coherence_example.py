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

    # The client is configured to be incoherent on the first call.
    client = coherence_example.MockZAIClient(api_key="test_key_incoherent", incoherent_attempts=1)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # ACT
    # With max_retries=1, it will try the initial call and then retry once.
    final_answer = await coherence_example.ask_zai(question, client, max_retries=1)

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


async def test_ask_zai_handles_empty_question():
    """
    Given an empty or whitespace-only question,
    ask_zai should return an empty string without calling the API.
    """
    # ARRANGE
    client = coherence_example.MockZAIClient(api_key="test_key_empty")

    # ACT & ASSERT for empty string
    final_answer_empty = await coherence_example.ask_zai("", client)
    assert final_answer_empty == "", "Expected an empty string for an empty question."

    # ACT & ASSERT for whitespace-only string
    final_answer_whitespace = await coherence_example.ask_zai("   ", client)
    assert final_answer_whitespace == "", "Expected an empty string for a whitespace-only question."


async def test_ask_zai_skips_repompt_for_coherent_response(mocker):
    """
    Given a coherent first response from the client,
    ask_zai should not re-prompt and should return the first answer.
    """
    # ARRANGE
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')

    # Configure the client to be coherent from the start.
    client = coherence_example.MockZAIClient(api_key="test_key_coherent", incoherent_attempts=0)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # ACT
    final_answer = await coherence_example.ask_zai(question, client)

    # ASSERT
    # 1. Check that the API was called only once.
    assert spy_create_async.call_count == 1, "Expected only one call to the API for a coherent response."

    # 2. Check that the final answer is the coherent one from the first call.
    assert "Esta é uma resposta coerente" in final_answer
    assert question in final_answer


async def test_ask_zai_handles_multiple_incoherent_responses(mocker):
    """
    Given a client that provides multiple incoherent responses,
    ask_zai should retry until it receives a coherent one.
    """
    # ARRANGE
    spy_create_async = mocker.spy(coherence_example.MockCompletions, 'create_async')

    # Configure the client to be incoherent for the first two attempts.
    client = coherence_example.MockZAIClient(api_key="test_key_multi_incoherent", incoherent_attempts=2)
    question = "What is the meaning of life?"

    # ACT
    # With max_retries=2, it will try the initial call and up to two retries.
    final_answer = await coherence_example.ask_zai(question, client, max_retries=2)

    # ASSERT
    # 1. Check that the API was called three times (initial + 2 retries).
    assert spy_create_async.call_count == 3, "Expected three calls to the API for two retries."

    # 2. Check that the final answer is the coherent one from the third call.
    assert "Claro, aqui está uma resposta coerente" in final_answer, "The final answer should be the coherent one from the third call."

    # 3. Inspect the messages for the third call to verify the full history.
    third_call_messages = spy_create_async.call_args_list[2].kwargs['messages']
    assert len(third_call_messages) == 6, "The third call should have 6 messages."
    assert third_call_messages[0]['role'] == 'system'
    assert third_call_messages[1]['role'] == 'user'
    assert third_call_messages[1]['content'] == question
    assert third_call_messages[2]['role'] == 'assistant' # First bad answer
    assert third_call_messages[3]['role'] == 'user'      # First re-prompt
    assert third_call_messages[4]['role'] == 'assistant' # Second bad answer
    assert third_call_messages[5]['role'] == 'user'      # Second re-prompt
    assert "reformule" in third_call_messages[5]['content']
