# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the SmartCalendarResolver environment."""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

StageName = Literal[
    "understand_request",
    "evaluate_availability",
    "propose_slot",
    "confirm_schedule",
]


class CalenderEnAction(Action):
    """Typed action for the deterministic scheduling workflow."""

    stage: StageName = Field(..., description="Current scheduling stage being executed")
    proposed_time_slot: Optional[str] = Field(
        default=None,
        description="Candidate slot proposed by the agent when proposing or confirming",
    )
    confirm_schedule: bool = Field(
        default=False,
        description="Whether the agent is explicitly confirming the selected schedule",
    )
    final_note: str = Field(
        default="",
        description="Short reasoning or confirmation note for the current stage",
    )


class CalenderEnObservation(Observation):
    """Observation for the SmartCalendarResolver workflow."""

    request: str = Field(default="", description="Scheduling request under consideration")
    availability: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Participant availability windows for the active scenario",
    )
    constraints: Dict[str, str] = Field(
        default_factory=dict,
        description="Scheduling constraints such as duration, priority, and deadline",
    )
    step_count: int = Field(default=0, description="Current number of steps taken")
    reward: float = Field(default=0.0, description="Reward assigned to the latest step")
    done: bool = Field(default=False, description="Whether the episode is complete")
    feedback: str = Field(default="", description="Environment feedback for the latest action")
    next_expected_stage: Optional[StageName] = Field(
        default="understand_request",
        description="Next valid stage expected by the environment",
    )
