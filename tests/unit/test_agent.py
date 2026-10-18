"""Agent state-machine tests with a scripted LLM (no network, fully deterministic)."""

from rootcite.domain.models import Alert
from rootcite.agent.toolkit import SUBMIT_DIAGNOSIS

from support import ScriptedLLM, submit, text, tool_use

ALERT = Alert("t-1", "checkout-api", "p99 latency above 2s and 5xx errors rising")


def evidence_ids(diagnosis, kind):
    return [e.evidence_id for e in diagnosis.evidence if e.kind == kind]


def test_happy_path_tool_loop_then_grounded_diagnosis(make_container):
    llm = ScriptedLLM(
        [
            tool_use(("list_recent_deploys", {"service": "checkout-api", "hours": 2})),
            tool_use(
                ("get_metric", {"service": "checkout-api", "metric": "db_connections_in_use"})
            ),
            submit(citations=["E1", "E6", "E7"]),
        ]
    )
    d = make_container(llm).agent.run(ALERT)

    assert d.tools_called == ["list_recent_deploys", "get_metric"]
    assert d.grounded and d.citations == ["E1", "E6", "E7"]
    assert not d.needs_human and d.confidence == 0.9
    assert len(evidence_ids(d, "knowledge_base")) == 5
    # Every tool_use must be answered by a tool_result in the next user turn.
    second_request = llm.calls[1]["messages"]
    assert second_request[-1]["content"][0]["type"] == "tool_result"


def test_tool_output_is_wrapped_as_untrusted(make_container):
    llm = ScriptedLLM(
        [tool_use(("search_logs", {"service": "checkout-api", "query": "coupon"})), submit()]
    )
    make_container(llm).agent.run(ALERT)
    tool_result = llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert "<untrusted_data" in tool_result and 'injection_suspected="true"' in tool_result


def test_prompt_injection_in_logs_escalates_to_human(make_container):
    llm = ScriptedLLM(
        [
            tool_use(("search_logs", {"service": "checkout-api", "query": "coupon"})),
            submit(citations=["E1", "E6"]),
        ]
    )
    d = make_container(llm).agent.run(ALERT)
    assert d.needs_human
    assert any("prompt injection" in w for w in d.warnings)


def test_side_effect_tools_are_blocked(make_container):
    llm = ScriptedLLM(
        [
            tool_use(("rollback_deploy", {"service": "checkout-api", "to_version": "v2.13.2"})),
            submit(citations=["E1"]),
        ]
    )
    d = make_container(llm).agent.run(ALERT)
    expected = 'rollback_deploy({"service": "checkout-api", "to_version": "v2.13.2"})'
    assert d.blocked_actions == [expected]
    assert "rollback_deploy" not in d.tools_called
    assert d.needs_human
    result = llm.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "BLOCKED" in result["content"]


def test_step_budget_forces_structured_submit(make_container):
    loop = [tool_use(("search_logs", {"service": "checkout-api", "query": "error"}))] * 6
    llm = ScriptedLLM([*loop, submit(citations=["E1", "E6"])])
    d = make_container(llm).agent.run(ALERT)

    final = llm.calls[-1]
    assert final["tool_choice"] == {"type": "tool", "name": SUBMIT_DIAGNOSIS}
    assert final["messages"][-1]["role"] == "user"
    assert d.grounded


def test_text_reply_triggers_forced_submit(make_container):
    llm = ScriptedLLM([text("Probably the database."), submit(citations=["E1"])])
    d = make_container(llm).agent.run(ALERT)
    assert llm.calls[1]["tool_choice"]["name"] == SUBMIT_DIAGNOSIS
    assert [m["role"] for m in llm.calls[1]["messages"]] == ["user", "assistant", "user"]
    assert d.grounded


def test_invalid_tool_arguments_return_error_to_model(make_container):
    llm = ScriptedLLM([tool_use(("get_metric", {"service": "checkout-api"})), submit()])
    d = make_container(llm).agent.run(ALERT)
    result = llm.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "metric" in result["content"]
    assert not d.grounded  # submit() default cites nothing


def test_no_submission_falls_back_safely(make_container):
    llm = ScriptedLLM([text("not sure"), text("still not sure")])
    d = make_container(llm).agent.run(ALERT)
    assert d.needs_human and not d.grounded and d.confidence == 0.0
