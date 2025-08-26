import pytest
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

    # 3. Inspect the messages for each call to be sure the logic is correct.
    first_call_messages = spy_create_async.call_args_list[0].kwargs['messages']
    second_call_messages = spy_create_async.call_args_list[1].kwargs['messages']

    # The first call should contain the original question.
    assert first_call_messages[-1]['content'] == question, "The first call should be the original question."

    # The second call should contain the re-prompt instruction.
    assert "reformule" in second_call_messages[-1]['content'], "The second call should contain the re-prompt instruction."
    assert question in second_call_messages[-1]['content'], "The second call should contain the original question for context."


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
