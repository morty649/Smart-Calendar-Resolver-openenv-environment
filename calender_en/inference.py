"""Deterministic baseline for the SmartCalendarResolver environment."""

from typing import List

try:
    from calender_en.models import CalenderEnAction
    from calender_en.server.calender_en_environment import CalenderEnEnvironment
except ModuleNotFoundError:
    from models import CalenderEnAction
    from server.calender_en_environment import CalenderEnEnvironment

TASK_NAME = "smart_calendar_resolution"
ENV_NAME = "calender_en"
MODEL_NAME = "deterministic-baseline"


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

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")

    try:
        env.reset()
        for action in _policy():
            steps += 1
            error = "null"
            try:
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
