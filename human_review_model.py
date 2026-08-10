from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HumanReviewParameters:
    detection_probability: float = 0.85
    review_seconds: float = 120.0
    # Hypothetical stochastic review-demand distribution. ``review_seconds`` is
    # the arithmetic mean; CV is held constant when review_time_scale changes.
    review_time_cv: float = 0.35
    review_time_scale: float = 1.0
    incorrect_override_probability: float = 0.01
    default_acceptance_effect: float = 0.25
    fatigue_sensitivity: float = 0.15
    staff_count: int = 5
    staffing_factor: float = 1.0
    shift_hours: float = 8.0
    # Normalised five-reviewer team scenario.  This absolute rate is not an
    # observed staffing claim; CAT112/112CV inform relative demand and mix.
    reference_arrivals_per_hour: float = 80.0
    explanation_time_multiplier: float = 1.0

    @property
    def effective_staff_count(self) -> int:
        return max(1, round(self.staff_count * self.staffing_factor))

    def with_changes(self, **kwargs) -> "HumanReviewParameters":
        return replace(self, **kwargs)


EXPLANATIONS = {
    "E0": {
        "name": "recommendation only",
        "extra_seconds": 0.0,
        "information_fraction": 0.25,
        "detection_multiplier": 0.68,
    },
    "E1": {
        "name": "recommendation + confidence",
        "extra_seconds": 8.0,
        "information_fraction": 0.45,
        "detection_multiplier": 0.76,
    },
    "E2": {
        "name": "recommendation + structured critical facts",
        "extra_seconds": 25.0,
        "information_fraction": 0.75,
        "detection_multiplier": 0.92,
    },
    "E3": {
        "name": "recommendation + structured justification",
        "extra_seconds": 50.0,
        "information_fraction": 0.90,
        "detection_multiplier": 1.00,
    },
    "E4": {
        "name": "uncertainty + missing-information warning + justification",
        "extra_seconds": 80.0,
        "information_fraction": 1.00,
        "detection_multiplier": 1.10,
    },
}


DECISION_WINDOWS_SECONDS = {"REV0": 900.0, "REV1": 480.0, "REV2": 240.0}
