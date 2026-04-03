"""Deterministic SmartCalendarResolver environment implementation."""

from typing import Dict, List, Optional, TypedDict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CalenderEnAction, CalenderEnObservation, StageName
except ImportError:
    from models import CalenderEnAction, CalenderEnObservation, StageName


class Scenario(TypedDict):
    difficulty: str
    request: str
    participants: List[str]
    availability: Dict[str, List[str]]
    constraints: Dict[str, str]
    ground_truth: str


SCENARIOS: List[Scenario] = [
    {
        "difficulty": "easy",
        "request": "Schedule a 30 minute kickoff between Alex and Priya before April 9.",
        "participants": ["Alex", "Priya"],
        "availability": {
            "Alex": [
                "2026-04-08 10:00-10:30 UTC",
                "2026-04-08 14:00-14:30 UTC",
            ],
            "Priya": [
                "2026-04-08 10:00-10:30 UTC",
                "2026-04-08 16:00-16:30 UTC",
            ],
        },
        "constraints": {
            "duration": "30 minutes",
            "priority": "kickoff meeting should happen at the earliest common slot",
            "deadline": "2026-04-09 18:00 UTC",
        },
        "ground_truth": "2026-04-08 10:00-10:30 UTC",
    },
    {
        "difficulty": "medium",
        "request": "Book a 45 minute design review for Mei, Jordan, and Sam before April 12 noon.",
        "participants": ["Mei", "Jordan", "Sam"],
        "availability": {
            "Mei": [
                "2026-04-10 13:00-13:45 UTC",
                "2026-04-11 09:00-09:45 UTC",
            ],
            "Jordan": [
                "2026-04-10 13:00-13:45 UTC",
                "2026-04-11 11:00-11:45 UTC",
            ],
            "Sam": [
                "2026-04-10 13:00-13:45 UTC",
                "2026-04-11 09:00-09:45 UTC",
            ],
        },
        "constraints": {
            "duration": "45 minutes",
            "priority": "use the earliest shared slot that avoids missing the review deadline",
            "deadline": "2026-04-12 12:00 UTC",
        },
        "ground_truth": "2026-04-10 13:00-13:45 UTC",
    },
    {
        "difficulty": "hard",
        "request": "Find a 60 minute executive sync for Elena, Ravi, Noor, and Luis before April 15 15:00 UTC.",
        "participants": ["Elena", "Ravi", "Noor", "Luis"],
        "availability": {
            "Elena": [
                "2026-04-14 08:00-09:00 UTC",
                "2026-04-15 09:00-10:00 UTC",
            ],
            "Ravi": [
                "2026-04-14 08:00-09:00 UTC",
                "2026-04-15 14:00-15:00 UTC",
            ],
            "Noor": [
                "2026-04-14 08:00-09:00 UTC",
                "2026-04-14 16:00-17:00 UTC",
            ],
            "Luis": [
                "2026-04-14 08:00-09:00 UTC",
                "2026-04-15 09:00-10:00 UTC",
            ],
        },
        "constraints": {
            "duration": "60 minutes",
            "priority": "executive sync must include every participant and happen before the deadline",
            "deadline": "2026-04-15 15:00 UTC",
        },
        "ground_truth": "2026-04-14 08:00-09:00 UTC",
    },
]

STAGES: List[StageName] = [
    "understand_request",
    "evaluate_availability",
    "propose_slot",
    "confirm_schedule",
]


class CalenderEnEnvironment(Environment):
    """A deterministic multi-step scheduling environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._scenario: Optional[Scenario] = None
        self._completed_stages: List[StageName] = []
        self._history: List[Dict[str, object]] = []
        self._selected_slot: Optional[str] = None
        self._done = False
        self._solved = False

    def reset(self) -> CalenderEnObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = SCENARIOS[self._reset_count % len(SCENARIOS)]
        self._reset_count += 1
        self._completed_stages = []
        self._history = []
        self._selected_slot = None
        self._done = False
        self._solved = False
        return self._observation(
            reward=0.0,
            done=False,
            feedback=(
                f"Loaded {self._scenario['difficulty']} scheduling scenario for "
                f"{', '.join(self._scenario['participants'])}."
            ),
            next_expected_stage=STAGES[0],
        )

    def step(self, action: CalenderEnAction) -> CalenderEnObservation:  # type: ignore[override]
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before step().")

        self._state.step_count += 1
        expected_stage = self._expected_stage()
        if expected_stage is None:
            return self._observation(
                reward=-1.0,
                done=True,
                feedback="Episode is already complete. Call reset() to start a new scenario.",
                next_expected_stage=None,
            )

        if action.stage in self._completed_stages:
            return self._observation(
                reward=-1.5,
                done=self._done,
                feedback=f"Stage '{action.stage}' was already completed. Repeated actions are penalized.",
                next_expected_stage=expected_stage,
            )

        if action.stage != expected_stage:
            return self._observation(
                reward=-1.0,
                done=False,
                feedback=f"Invalid stage order. Expected '{expected_stage}' next.",
                next_expected_stage=expected_stage,
            )

        handler = getattr(self, f"_handle_{action.stage}")
        reward, feedback = handler(action)
        self._history.append(
            {
                "stage": action.stage,
                "proposed_time_slot": action.proposed_time_slot,
                "confirm_schedule": action.confirm_schedule,
                "reward": reward,
                "feedback": feedback,
            }
        )
        self._completed_stages.append(action.stage)
        next_stage = self._expected_stage()
        self._done = next_stage is None
        return self._observation(
            reward=reward,
            done=self._done,
            feedback=feedback,
            next_expected_stage=next_stage,
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def history(self) -> List[Dict[str, object]]:
        return list(self._history)

    @property
    def solved(self) -> bool:
        return self._solved

    def _expected_stage(self) -> Optional[StageName]:
        if len(self._completed_stages) >= len(STAGES):
            return None
        return STAGES[len(self._completed_stages)]

    def _handle_understand_request(self, action: CalenderEnAction) -> tuple[float, str]:
        note = action.final_note.lower()
        if action.proposed_time_slot or action.confirm_schedule:
            return -0.5, "Understanding stage should not propose or confirm a schedule yet."
        if "participant" in note or "deadline" in note or "objective" in note:
            return 1.0, "Request understood. Participants and deadline were identified correctly."
        return 0.5, "Request acknowledged. More explicit mention of participants or deadline would be better."

    def _handle_evaluate_availability(self, action: CalenderEnAction) -> tuple[float, str]:
        note = action.final_note.lower()
        if action.proposed_time_slot or action.confirm_schedule:
            return -0.5, "Availability stage should evaluate options before proposing a slot."
        common_slots = self._common_slots()
        if not common_slots:
            return -2.0, "Scenario is misconfigured because no common slots exist."
        if "earliest" in note or "intersect" in note or "shared" in note:
            return 1.5, f"Availability evaluated. Common slot candidates: {', '.join(common_slots)}."
        return 1.0, f"Availability checked. Common slot candidates: {', '.join(common_slots)}."

    def _handle_propose_slot(self, action: CalenderEnAction) -> tuple[float, str]:
        proposed_slot = action.proposed_time_slot
        if not proposed_slot:
            return -2.0, "A proposed_time_slot is required during the propose_slot stage."
        if action.confirm_schedule:
            return -0.5, "Do not confirm the meeting before the confirmation stage."
        reward = 0.0
        feedback_parts: List[str] = []
        if proposed_slot not in self._common_slots():
            reward -= 2.0
            feedback_parts.append("Proposed slot is not available for every participant.")
        else:
            reward += 1.5
            feedback_parts.append("Proposed slot satisfies shared availability.")
        if proposed_slot == self._scenario["ground_truth"]:
            reward += 2.0
            feedback_parts.append("Proposed slot matches the correct deterministic solution.")
            self._selected_slot = proposed_slot
        else:
            feedback_parts.append(
                f"Correct slot is {self._scenario['ground_truth']} based on earliest valid availability."
            )
            self._selected_slot = proposed_slot
        if "earliest" in action.final_note.lower() or "deadline" in action.final_note.lower():
            reward += 0.5
            feedback_parts.append("Proposal note reflects the deadline and priority constraints.")
        return reward, " ".join(feedback_parts)

    def _handle_confirm_schedule(self, action: CalenderEnAction) -> tuple[float, str]:
        proposed_slot = action.proposed_time_slot or self._selected_slot
        if not action.confirm_schedule:
            return -2.0, "Confirmation stage requires confirm_schedule=True."
        if not proposed_slot:
            return -1.5, "Confirmation requires a concrete time slot."
        if proposed_slot != self._scenario["ground_truth"]:
            return -1.0, "Confirmation used the wrong slot."
        reward = 2.0
        note = action.final_note.lower()
        if "confirmed" in note or "invite" in note:
            reward += 1.0
            feedback = "Schedule confirmed with a valid final note."
        else:
            feedback = "Schedule confirmed, but the final note should explicitly mention confirmation."
        self._selected_slot = proposed_slot
        self._solved = True
        return reward, feedback

    def _common_slots(self) -> List[str]:
        if self._scenario is None:
            return []
        participant_slots = [
            set(self._scenario["availability"][participant])
            for participant in self._scenario["participants"]
        ]
        common = set.intersection(*participant_slots)
        return sorted(common)

    def _observation(
        self,
        reward: float,
        done: bool,
        feedback: str,
        next_expected_stage: Optional[StageName],
    ) -> CalenderEnObservation:
        if self._scenario is None:
            raise RuntimeError("Scenario is not initialized.")
        return CalenderEnObservation(
            request=self._scenario["request"],
            availability=self._scenario["availability"],
            constraints=self._scenario["constraints"],
            step_count=self._state.step_count,
            reward=reward,
            done=done,
            feedback=feedback,
            next_expected_stage=next_expected_stage,
        )
