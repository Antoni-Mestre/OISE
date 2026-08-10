from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuthorityRegime:
    code: str
    name: str
    ai_information_role: str
    ai_recommendation_role: str
    default_action: str
    human_confirmation_required: str
    automatic_execution: str
    veto_possible: str
    mandatory_escalation: str
    final_authority_structure: str
    traceability_design: str
    default_ai: bool
    recommendation_presented: bool
    veto_window_seconds: float | None = None


REGIMES = {
    "R0": AuthorityRegime(
        "R0", "Human-centred decision support", "Extracts facts and drafts documentation",
        "No action recommendation shown", "None", "Yes, all consequential decisions",
        "No", "Not applicable", "No automatic delegation", "Human",
        "All simulated events logged", False, False,
    ),
    "R1": AuthorityRegime(
        "R1", "Human decision with AI recommendation", "Facts, confidence and rationale",
        "Advisory; explicit human choice", "No pre-selected action", "Yes, every decision",
        "No", "Not applicable", "No automatic delegation", "Human",
        "All simulated events logged", False, True,
    ),
    "R2": AuthorityRegime(
        "R2", "AI default / human confirmation", "Facts, confidence and rationale",
        "Pre-selected recommendation", "AI recommendation", "Yes, every decision",
        "No", "Modification/rejection before confirmation", "Not applicable", "Human formal; AI default",
        "All simulated events logged", True, True,
    ),
    "R3": AuthorityRegime(
        "R3", "Selective automation", "Facts, confidence and rationale",
        "Recommendation plus confidence threshold", "AI for delegated cases", "Only escalated cases",
        "Low-risk/high-confidence cases", "For escalated cases", "S2, ambiguity or low confidence",
        "Mixed by routing result", "All simulated events logged", False, True,
    ),
    "R4": AuthorityRegime(
        "R4", "AI action with human veto", "Facts, confidence and rationale",
        "Action unless vetoed", "AI action", "No", "All cases after veto window", "Yes",
        "No mandatory escalation", "AI effective; human nominal veto",
        "All simulated events logged", True, True, 180.0,
    ),
    "R5": AuthorityRegime(
        "R5", "Policy-governed delegation", "Facts, uncertainty and rule trace",
        "Within explicit delegation boundary", "AI only inside policy boundary", "Escalated cases only",
        "Routine cases inside policy boundary", "For escalated cases",
        "S2, ambiguity, low confidence and protected fire/rescue classes",
        "Explicitly allocated by policy rule", "Delegation decision and authority holder logged", False, True,
    ),
    "RP": AuthorityRegime(
        "RP", "Universal human review with risk-priority queue", "Facts, confidence and rationale",
        "Advisory; explicit human choice", "No pre-selected action", "Yes, every decision",
        "No", "Not applicable", "No automatic delegation", "Human",
        "Risk-priority queue decisions and all review events logged", False, True,
    ),
}


def review_required(
    regime_code: str,
    severity: np.ndarray,
    confidence: np.ndarray,
    ambiguous: np.ndarray,
    incident_type: np.ndarray,
    selective_threshold: float = 0.80,
    governed_threshold: float = 0.72,
    governance_variant: str = "full",
) -> np.ndarray:
    n = len(severity)
    if regime_code in {"R0", "R1", "R2", "R4", "RP"}:
        return np.ones(n, dtype=bool)
    if regime_code == "R3":
        delegated = (severity == "S0") & (confidence >= selective_threshold) & (~ambiguous)
        return ~delegated
    if regime_code == "R5":
        protected = np.isin(incident_type, ["fire", "rescue"])
        components = {
            "confidence_only": confidence < governed_threshold,
            "severity_only": severity == "S2",
            "mandatory_only": (severity == "S2") | ambiguous | protected,
            "confidence_severity": (severity == "S2") | (confidence < governed_threshold),
            "full": (severity == "S2") | ambiguous | (confidence < governed_threshold) | protected,
        }
        if governance_variant not in components:
            raise KeyError(f"Unknown R5 governance variant: {governance_variant}")
        return components[governance_variant]
    raise KeyError(regime_code)


def nominal_human_responsibility(regime_code: str, review_mask: np.ndarray) -> np.ndarray:
    if regime_code == "R5":
        return review_mask.copy()
    return np.ones(len(review_mask), dtype=bool)
