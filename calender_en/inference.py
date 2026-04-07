"""Proxy-aware inference entrypoint for the SmartCalendarResolver environment."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

from openai import OpenAI

try:
    from calender_en.client import CalenderEnEnv
    from calender_en.models import CalenderEnAction, CalenderEnObservation
    from calender_en.server.calender_en_environment import CalenderEnEnvironment
except ModuleNotFoundError:
    from client import CalenderEnEnv
    from models import CalenderEnAction, CalenderEnObservation
    from server.calender_en_environment import CalenderEnEnvironment

TASK_NAME = "smart_calendar_resolution"
ENV_NAME = "calender_en"
DEFAULT_MODEL_NAME = "deterministic-baseline"


@dataclass
class InferenceConfig:
    env_base_url: str = field(default_factory=lambda: os.getenv("ENV_BASE_URL", ""))
    llm_api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    )


@dataclass
class StepOutcome:
    observation: CalenderEnObservation
    reward: float
    done: bool


class LocalEnvRunner:
    def __init__(self) -> None:
        self._env = CalenderEnEnvironment()

    def reset(self) -> CalenderEnObservation:
        return self._env.reset()

    def step(self, action: CalenderEnAction) -> StepOutcome:
        observation = self._env.step(action)
        return StepOutcome(
            observation=observation,
            reward=float(observation.reward),
            done=bool(observation.done),
        )

    def close(self) -> None:
        return None


class RemoteEnvRunner:
    def __init__(self, base_url: str) -> None:
        self._client = CalenderEnEnv(base_url=base_url)

    def reset(self) -> CalenderEnObservation:
        return self._client.reset().observation

    def step(self, action: CalenderEnAction) -> StepOutcome:
        result = self._client.step(action)
        return StepOutcome(
            observation=result.observation,
            reward=float(result.reward or 0.0),
            done=bool(result.done),
        )

    def close(self) -> None:
        self._client.close()


def _runner(config: InferenceConfig) -> LocalEnvRunner | RemoteEnvRunner:
    if config.env_base_url:
        return RemoteEnvRunner(config.env_base_url)
    return LocalEnvRunner()


def _should_use_llm_proxy(config: InferenceConfig) -> bool:
    return bool(config.llm_api_base_url or config.llm_api_key)


def _create_proxy_client(config: InferenceConfig) -> OpenAI:
    missing = [
        name
        for name, value in (
            ("API_BASE_URL", config.llm_api_base_url),
            ("API_KEY", config.llm_api_key),
            ("MODEL_NAME", config.model_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required hackathon environment variables: "
            + ", ".join(missing)
        )
    return OpenAI(base_url=config.llm_api_base_url, api_key=config.llm_api_key)


def _common_slots(observation: CalenderEnObservation) -> List[str]:
    participant_slots = [set(slots) for slots in observation.availability.values()]
    if not participant_slots:
        raise ValueError("Observation does not include participant availability.")
    return sorted(set.intersection(*participant_slots))


def _parse_slot_start(slot: str) -> datetime:
    return datetime.strptime(slot[:16], "%Y-%m-%d %H:%M")


def _parse_deadline(deadline: str) -> datetime:
    normalized = deadline.replace(" UTC", "")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported deadline format: {deadline}")


def _select_slot(observation: CalenderEnObservation) -> str:
    common_slots = _common_slots(observation)
    deadline = observation.constraints.get("deadline")
    if deadline:
        cutoff = _parse_deadline(deadline)
        valid_slots = [slot for slot in common_slots if _parse_slot_start(slot) <= cutoff]
        if valid_slots:
            return valid_slots[0]
    return common_slots[0]


def _fallback_note(stage: str) -> str:
    notes = {
        "understand_request": "Identify the meeting objective, participants, and deadline.",
        "evaluate_availability": "Intersect participant availability and filter slots before the deadline.",
        "propose_slot": "Choose the earliest common 30 minute slot before the deadline.",
        "confirm_schedule": "Confirmed with all participants and calendar invite is ready.",
    }
    return notes[stage]


def _planned_action(observation: CalenderEnObservation) -> CalenderEnAction:
    stage = observation.next_expected_stage
    if stage is None:
        raise ValueError("No next stage available.")

    if stage == "understand_request":
        return CalenderEnAction(stage=stage, final_note=_fallback_note(stage))

    if stage == "evaluate_availability":
        return CalenderEnAction(stage=stage, final_note=_fallback_note(stage))

    slot = _select_slot(observation)
    if stage == "propose_slot":
        return CalenderEnAction(
            stage=stage,
            proposed_time_slot=slot,
            final_note=_fallback_note(stage),
        )

    return CalenderEnAction(
        stage=stage,
        proposed_time_slot=slot,
        confirm_schedule=True,
        final_note=_fallback_note(stage),
    )


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
    config: InferenceConfig,
    observation: CalenderEnObservation,
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
        model=config.model_name,
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
    config = InferenceConfig()
    env = _runner(config)
    rewards: List[str] = []
    steps = 0
    success = False
    client = _create_proxy_client(config) if _should_use_llm_proxy(config) else None

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={config.model_name}")

    try:
        observation = env.reset()
        while not observation.done and observation.next_expected_stage is not None:
            planned_action = _planned_action(observation)
            action = planned_action
            steps += 1
            try:
                if client is not None:
                    action = _generate_action_with_proxy(
                        client, config, observation, planned_action
                    )
                outcome = env.step(action)
                observation = outcome.observation
                reward_text = f"{outcome.reward:.2f}"
                rewards.append(reward_text)
                print(
                    f"[STEP] step={steps} action={_format_action(action)} "
                    f"reward={reward_text} done={str(outcome.done).lower()} error=null"
                )
                success = outcome.done
            except Exception as exc:
                rewards.append("0.00")
                print(
                    f"[STEP] step={steps} action={_format_action(action)} "
                    f"reward=0.00 done=false error={str(exc)}"
                )
                success = False
                break
    except Exception:
        success = False
    finally:
        env.close()
        print(
            f"[END] success={str(success).lower()} steps={steps} rewards={','.join(rewards)}"
        )


if __name__ == "__main__":
    main()
