"""OpenAI-compatible adapter: format translation, error mapping, and a full agent run."""

import json

import pytest

from rootcite.adapters.openai_compat import (
    OpenAICompatLLM,
    from_openai_response,
    parse_arguments,
    to_openai_messages,
    to_openai_tool_choice,
    to_openai_tools,
)
from rootcite.domain.errors import LLMRequestError, RetryableError
from rootcite.domain.models import Alert


class FakeTransport:
    """Records requests and replays canned (status, body) responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "payload": json.loads(body)})
        status, data = self.responses.pop(0)
        return status, json.dumps(data).encode() if isinstance(data, dict) else data


def tool_call_response(*calls, prompt=120, completion=30):
    return 200, {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                    for i, (name, args) in enumerate(calls)
                ],
            },
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def test_conversation_translates_to_openai_format():
    history = [
        {"role": "user", "content": "New alert"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking deploys."},
            {"type": "tool_use", "id": "c1", "name": "list_recent_deploys",
             "input": {"service": "a"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "v2 deployed"},
            {"type": "text", "text": "Budget reached. Submit now."},
        ]},
    ]
    out = to_openai_messages("rules", history)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "user"]
    assert out[2]["tool_calls"][0]["function"] == {
        "name": "list_recent_deploys", "arguments": '{"service": "a"}'}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "v2 deployed"}
    assert out[4]["content"] == "Budget reached. Submit now."


def test_tool_errors_are_marked_and_tools_and_choice_are_mapped():
    out = to_openai_messages("s", [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "x", "content": "BLOCKED", "is_error": True}]}])
    assert out[1]["content"] == "ERROR: BLOCKED"
    schema = {"type": "object"}
    tools = to_openai_tools([{"name": "t", "description": "d", "input_schema": schema}])
    assert tools == [{"type": "function",
                      "function": {"name": "t", "description": "d", "parameters": schema}}]
    assert to_openai_tool_choice({"type": "tool", "name": "submit_diagnosis"}) == {
        "type": "function", "function": {"name": "submit_diagnosis"}}
    assert to_openai_tool_choice(None) is None


def test_responses_parse_into_tool_calls_and_strip_thinking():
    r = from_openai_response(tool_call_response(("get_metric", {"service": "a", "metric": "m"}))[1])
    assert r.tool_calls[0].name == "get_metric" and r.tool_calls[0].arguments["metric"] == "m"
    assert r.assistant_content[0]["type"] == "tool_use"
    assert r.usage.input_tokens == 120 and r.usage.output_tokens == 30
    text = from_openai_response({"choices": [{"message": {"content": "<think>hmm</think> Done."}}]})
    assert text.text == "Done." and text.tool_calls == []


def test_malformed_arguments_become_a_rejectable_field():
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("{not json") == {"_unparseable_arguments": "{not json"}
    assert parse_arguments("[1, 2]") == {"_unparseable_arguments": "[1, 2]"}


@pytest.mark.parametrize("status, error", [(429, RetryableError), (503, RetryableError),
                                          (401, LLMRequestError), (404, LLMRequestError)])
def test_http_errors_map_to_domain_errors(status, error):
    llm = OpenAICompatLLM("m", "k", transport=FakeTransport([(status, b'{"error": "x"}')]))
    with pytest.raises(error):
        llm.complete(system="s", messages=[{"role": "user", "content": "hi"}], tools=[])


def test_request_carries_model_key_tools_and_forced_choice():
    transport = FakeTransport([tool_call_response(("submit_diagnosis", {}))])
    llm = OpenAICompatLLM("openai/gpt-oss-20b", "secret-key", base_url="https://api.example/v1/",
                          transport=transport)
    llm.complete(system="s", messages=[{"role": "user", "content": "hi"}],
                 tools=[{"name": "submit_diagnosis", "input_schema": {"type": "object"}}],
                 tool_choice={"type": "tool", "name": "submit_diagnosis"})
    req = transport.requests[0]
    assert req["url"] == "https://api.example/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer secret-key"
    assert req["payload"]["model"] == "openai/gpt-oss-20b"
    assert req["payload"]["tool_choice"]["function"]["name"] == "submit_diagnosis"


def test_full_agent_run_through_the_adapter(make_container):
    """The real agent, tools, guardrails, and verifier, with the adapter in the loop."""
    transport = FakeTransport([
        tool_call_response(("list_recent_deploys", {"service": "checkout-api", "hours": 6}),
                           ("search_logs", {"service": "checkout-api", "query": "pool"})),
        tool_call_response(("submit_diagnosis", {
            "root_cause": "Deploy v2.14.0 cut the DB pool from 20 to 5 [E6].",
            "confidence": 0.9, "citations": ["E1", "E6", "E7"],
            "recommended_actions": ["Roll back v2.14.0"], "needs_human": False})),
    ])
    container = make_container(OpenAICompatLLM("m", "k", transport=transport))
    alert = Alert("a1", "checkout-api", "p99 latency above 2s and 5xx errors rising")
    d = container.agent.run(alert)

    assert d.tools_called == ["list_recent_deploys", "search_logs"]
    assert d.grounded and set(d.citations) == {"E1", "E6", "E7"}
    second = transport.requests[1]["payload"]["messages"]
    assert [m["role"] for m in second[-3:]] == ["assistant", "tool", "tool"]
    assert "<untrusted_data" in second[-1]["content"]
