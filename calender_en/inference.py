"""Deterministic baseline for the SmartCalendarResolver environment."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

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


@dataclass
class InferenceConfig:
    api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "deterministic-baseline"))
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    model_base_url: str = field(
        default_factory=lambda: os.getenv("MODEL_BASE_URL", "https://router.huggingface.co/v1")
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
            reward=observation.reward,
            done=observation.done,
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


def _build_openai_client(config: InferenceConfig) -> Optional[OpenAI]:
    if not config.hf_token:
        return None
    return OpenAI(base_url=config.model_base_url, api_key=config.hf_token)


def _common_slots(observation: CalenderEnObservation) -> List[str]:
    participant_slots = [set(slots) for slots in observation.availability.values()]
    if not participant_slots:
        raise ValueError("Observation does not include participant availability.")
    return sorted(set.intersection(*participant_slots))


def _parse_slot_start(slot: str) -> datetime:
    return datetime.strptime(slot[:16], "%Y-%m-%d %H:%M")


def _parse_deadline(deadline: str) -> datetime:
    normalized = deadline.replace(" UTC", "")
    formats = ("%Y-%m-%d %H:%M", "%Y-%m-%d")
    for fmt in formats:
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


def _remote_note(
    client: OpenAI,
    config: InferenceConfig,
    observation: CalenderEnObservation,
    stage: str,
    slot: Optional[str],
) -> str:
    prompt = (
        "You are generating a short scheduling-agent note. "
        f"Stage: {stage}. Request: {observation.request}. "
        f"Constraints: {observation.constraints}. "
        f"Shared slot: {slot or 'none yet'}. "
        "Return one sentence only."
    )
    try:
        response = client.responses.create(
            model=config.model_name,
            input=prompt,
            max_output_tokens=40,
        )
        text = response.output_text.strip()
        if text:
            return text
    except Exception:
        pass

    completion = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=40,
    )
    text = completion.choices[0].message.content or ""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model.")
    return text


def _note_for_stage(
    client: Optional[OpenAI],
    config: InferenceConfig,
    observation: CalenderEnObservation,
    stage: str,
    slot: Optional[str] = None,
) -> str:
    if client is None or config.model_name == "deterministic-baseline":
        return _fallback_note(stage)
    try:
        return _remote_note(client, config, observation, stage, slot)
    except Exception:
        return _fallback_note(stage)


def _next_action(
    observation: CalenderEnObservation,
    client: Optional[OpenAI],
    config: InferenceConfig,
) -> CalenderEnAction:
    stage = observation.next_expected_stage
    if stage is None:
        raise ValueError("No next stage available.")

    if stage == "understand_request":
        return CalenderEnAction(
            stage=stage,
            final_note=_note_for_stage(client, config, observation, stage),
        )

    if stage == "evaluate_availability":
        return CalenderEnAction(
            stage=stage,
            final_note=_note_for_stage(client, config, observation, stage),
        )

    slot = _select_slot(observation)
    if stage == "propose_slot":
        return CalenderEnAction(
            stage=stage,
            proposed_time_slot=slot,
            final_note=_note_for_stage(client, config, observation, stage, slot),
        )

    return CalenderEnAction(
        stage=stage,
        proposed_time_slot=slot,
        confirm_schedule=True,
        final_note=_note_for_stage(client, config, observation, stage, slot),
    )


def _format_action(action: CalenderEnAction) -> str:
    parts = [f"stage={action.stage}"]
    if action.proposed_time_slot:
        parts.append(f"slot={action.proposed_time_slot}")
    parts.append(f"confirm={str(action.confirm_schedule).lower()}")
    if action.final_note:
        parts.append(f"note={action.final_note}")
    return "|".join(parts)


def _runner(config: InferenceConfig) -> LocalEnvRunner | RemoteEnvRunner:
    if config.api_base_url:
        return RemoteEnvRunner(config.api_base_url)
    return LocalEnvRunner()


def main() -> None:
    config = InferenceConfig()
    client = _build_openai_client(config)
    env = _runner(config)
    rewards: List[str] = []
    steps = 0
    success = False

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={config.model_name}")

    try:
        observation = env.reset()
        while not observation.done and observation.next_expected_stage is not None:
            action = _next_action(observation, client, config)
            steps += 1
            try:
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
        print(f"[END] success={str(success).lower()} steps={steps} rewards={','.join(rewards)}")


if __name__ == "__main__":
    main()
