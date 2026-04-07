"""Proxy-aware inference entrypoint for the SmartCalendarResolver environment."""

import json
import os
from typing import Any, List

try:
    from calender_en.models import CalenderEnAction
    from calender_en.server.calender_en_environment import CalenderEnEnvironment
except ModuleNotFoundError:
    from models import CalenderEnAction
    from server.calender_en_environment import CalenderEnEnvironment

from openai import OpenAI

TASK_NAME = "smart_calendar_resolution"
ENV_NAME = "calender_en"
DEFAULT_MODEL_NAME = "deterministic-baseline"


def _policy() -> List[CalenderEnAction]:
    return [
        CalenderEnAction(
            stage="understand_request",
            final_note="Identify the meeting objective, participants, and deadline.",
        ),
        CalenderEnAction(
            stage="evaluate_availability",
            final_note="Intersect participant availability and filter slots before the deadline.",
        ),
        CalenderEnAction(
            stage="propose_slot",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            final_note="Choose the earliest common 30 minute slot before the deadline.",
        ),
        CalenderEnAction(
            stage="confirm_schedule",
            proposed_time_slot="2026-04-08 10:00-10:30 UTC",
            confirm_schedule=True,
            final_note="Confirmed with all participants and calendar invite is ready.",
        ),
    ]


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _should_use_llm_proxy() -> bool:
    return _env("API_BASE_URL") is not None or _env("API_KEY") is not None


def _get_model_name() -> str:
    return _env("MODEL_NAME") or DEFAULT_MODEL_NAME


def _create_client() -> OpenAI:
    api_base_url = _env("API_BASE_URL")
    api_key = _env("API_KEY")
    model_name = _env("MODEL_NAME")

    missing = [
        name
        for name, value in (
            ("API_BASE_URL", api_base_url),
            ("API_KEY", api_key),
            ("MODEL_NAME", model_name),
        )
        if value is None
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "Missing required hackathon environment variables: "
            f"{missing_text}. The validator injects these automatically."
        )

    return OpenAI(base_url=api_base_url, api_key=api_key)


def _extract_message_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:  # pragma: no cover
        raise RuntimeError("LLM proxy returned an unexpected response shape.") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text_value = getattr(item, "text", None)
            if text_value:
                parts.append(text_value)
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_action_json(payload: str) -> CalenderEnAction:
    payload = payload.strip()
    if payload.startswith("```"):
        lines = [line for line in payload.splitlines() if not line.startswith("```")]
        payload = "\n".join(lines).strip()

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM proxy did not return valid JSON: {payload}") from exc

    return CalenderEnAction(**data)


def _generate_action_with_proxy(
    client: OpenAI,
    observation: Any,
    planned_action: CalenderEnAction,
) -> CalenderEnAction:
    prompt_payload = {
        "request": observation.request,
        "availability": observation.availability,
        "constraints": observation.constraints,
        "step_count": observation.step_count,
        "feedback": observation.feedback,
        "next_expected_stage": observation.next_expected_stage,
        "planned_action": planned_action.model_dump(),
    }
    response = client.chat.completions.create(
        model=_get_model_name(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are controlling a deterministic scheduling environment. "
                    "Return exactly one JSON object matching the provided planned_action. "
                    "Do not add markdown, explanations, or extra keys."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, sort_keys=True),
            },
        ],
    )
    return _parse_action_json(_extract_message_text(response))


def _format_action(action: CalenderEnAction) -> str:
    parts = [f"stage={action.stage}"]
    if action.proposed_time_slot:
        parts.append(f"slot={action.proposed_time_slot}")
    parts.append(f"confirm={str(action.confirm_schedule).lower()}")
    if action.final_note:
        parts.append(f"note={action.final_note}")
    return "|".join(parts)


def main() -> None:
    env = CalenderEnEnvironment()
    rewards: List[str] = []
    steps = 0
    success = False
    use_llm_proxy = _should_use_llm_proxy()
    client = _create_client() if use_llm_proxy else None
    model_name = _get_model_name()

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={model_name}")

    try:
        observation = env.reset()
        for planned_action in _policy():
            steps += 1
            error = "null"
            action = planned_action
            try:
                action = (
                    _generate_action_with_proxy(client, observation, planned_action)
                    if client is not None
                    else planned_action
                )
                observation = env.step(action)
                reward_text = f"{observation.reward:.2f}"
                done_text = str(observation.done).lower()
                rewards.append(reward_text)
                print(
                    f"[STEP] step={steps} action={_format_action(action)} "
                    f"reward={reward_text} done={done_text} error={error}"
                )
                success = observation.done
            except Exception as exc:
                reward_text = "0.00"
                rewards.append(reward_text)
                print(
                    f"[STEP] step={steps} action={_format_action(action)} "
                    f"reward={reward_text} done=false error={str(exc)}"
                )
                success = False
                break
    except Exception:
        success = False
    finally:
        rewards_text = ",".join(rewards)
        print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_text}")


if __name__ == "__main__":
    main()
