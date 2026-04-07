from contextlib import redirect_stdout
from io import StringIO

from calender_en import inference


def test_inference_runs_end_to_end_without_crashing() -> None:
    output = StringIO()

    with redirect_stdout(output):
        inference.main()

    rendered = output.getvalue()
    lines = rendered.strip().splitlines()
    assert len(lines) == 6
    assert lines[0] == "[START] task=smart_calendar_resolution env=calender_en model=deterministic-baseline"
    assert lines[-1] == "[END] success=true steps=4 rewards=1.00,1.50,4.00,3.00"
    assert all(line.startswith("[STEP]") for line in lines[1:5])


def test_inference_output_is_deterministic() -> None:
    first = StringIO()
    second = StringIO()

    with redirect_stdout(first):
        inference.main()
    with redirect_stdout(second):
        inference.main()

    assert first.getvalue() == second.getvalue()


def test_inference_uses_hackathon_proxy_env_vars(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        class Choice:
            class Message:
                content = (
                    '{"stage":"understand_request","proposed_time_slot":null,'
                    '"confirm_schedule":false,'
                    '"final_note":"Identify the meeting objective, participants, and deadline."}'
                )

            message = Message()

        choices = [Choice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = FakeChat()

    monkeypatch.setenv("API_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("API_KEY", "proxy-test-key")
    monkeypatch.setenv("MODEL_NAME", "meta-hackathon-model")
    monkeypatch.setattr(inference, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        inference,
        "_policy",
        lambda: [
            inference.CalenderEnAction(
                stage="understand_request",
                final_note="Identify the meeting objective, participants, and deadline.",
            )
        ],
    )

    output = StringIO()
    with redirect_stdout(output):
        inference.main()

    rendered = output.getvalue().strip().splitlines()
    assert rendered[0] == (
        "[START] task=smart_calendar_resolution env=calender_en "
        "model=meta-hackathon-model"
    )
    assert captured["base_url"] == "https://proxy.example.com/v1"
    assert captured["api_key"] == "proxy-test-key"
    assert captured["request"]["model"] == "meta-hackathon-model"
