from __future__ import annotations

import heapq
from typing import overload

import numpy as np
import pandas as pd

from authority_regimes import REGIMES, nominal_human_responsibility, review_required
from human_review_model import DECISION_WINDOWS_SECONDS, EXPLANATIONS, HumanReviewParameters
from workload_model import sample_cases


def _fifo_schedule(
    arrivals: np.ndarray,
    review_mask: np.ndarray,
    service_seconds: np.ndarray,
    staff_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.full(len(arrivals), np.nan)
    ends = np.full(len(arrivals), np.nan)
    server = np.full(len(arrivals), -1, dtype=int)
    available = np.zeros(staff_count, dtype=float)
    for index in np.flatnonzero(review_mask):
        slot = int(np.argmin(available))
        start = max(float(arrivals[index]), float(available[slot]))
        end = start + float(service_seconds[index])
        starts[index] = start
        ends[index] = end
        server[index] = slot
        available[slot] = end
    return starts, ends, server


def _priority_schedule(
    arrivals: np.ndarray,
    review_mask: np.ndarray,
    service_seconds: np.ndarray,
    staff_count: int,
    priority: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.full(len(arrivals), np.nan)
    ends = np.full(len(arrivals), np.nan)
    server_assignment = np.full(len(arrivals), -1, dtype=int)
    review_indices = np.flatnonzero(review_mask)
    waiting: list[tuple[float, float, int]] = []
    servers: list[tuple[float, int]] = [(0.0, slot) for slot in range(staff_count)]
    heapq.heapify(servers)
    cursor = 0
    while cursor < len(review_indices) or waiting:
        server_time, slot = heapq.heappop(servers)
        if not waiting and cursor < len(review_indices):
            server_time = max(server_time, float(arrivals[review_indices[cursor]]))
        while cursor < len(review_indices) and arrivals[review_indices[cursor]] <= server_time:
            index = int(review_indices[cursor])
            heapq.heappush(waiting, (-float(priority[index]), float(arrivals[index]), index))
            cursor += 1
        if not waiting:
            heapq.heappush(servers, (server_time, slot))
            continue
        _, _, index = heapq.heappop(waiting)
        start = max(server_time, float(arrivals[index]))
        end = start + float(service_seconds[index])
        starts[index] = start
        ends[index] = end
        server_assignment[index] = slot
        heapq.heappush(servers, (end, slot))
    return starts, ends, server_assignment


def _queue_statistics(
    arrivals: np.ndarray,
    starts: np.ndarray,
    review_mask: np.ndarray,
    horizon: float,
) -> tuple[float, int]:
    events: list[tuple[float, int]] = []
    for index in np.flatnonzero(review_mask):
        arrival = float(arrivals[index])
        if arrival <= horizon:
            events.append((arrival, 1))
        if np.isfinite(starts[index]) and starts[index] <= horizon:
            events.append((float(starts[index]), -1))
    if not events:
        return 0.0, 0
    events.sort()
    queue = 0
    maximum = 0
    area = 0.0
    previous = 0.0
    cursor = 0
    while cursor < len(events):
        event_time = events[cursor][0]
        time = min(event_time, horizon)
        area += queue * max(0.0, time - previous)
        delta = 0
        while cursor < len(events) and events[cursor][0] == event_time:
            delta += events[cursor][1]
            cursor += 1
        queue = max(0, queue + delta)
        maximum = max(maximum, queue)
        previous = time
        if time >= horizon:
            break
    area += queue * max(0.0, horizon - previous)
    return float(area / horizon), int(maximum)


def _safe_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = values if mask is None else values[mask]
    return float(np.nanmean(selected)) if len(selected) else float("nan")


def _review_times(
    rng: np.random.Generator,
    n: int,
    params: HumanReviewParameters,
) -> np.ndarray:
    mean = params.review_seconds * params.review_time_scale
    cv = max(params.review_time_cv, 0.0)
    if cv == 0:
        return np.full(n, mean, dtype=float)
    sigma2 = np.log(cv * cv + 1.0)
    mu = np.log(mean) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n)


def _policy_reason(
    severity: np.ndarray,
    confidence: np.ndarray,
    ambiguous: np.ndarray,
    incident_type: np.ndarray,
    threshold: float,
) -> np.ndarray:
    reasons = []
    for sev, conf, amb, kind in zip(severity, confidence, ambiguous, incident_type):
        active = []
        if sev == "S2":
            active.append("high_consequence")
        if amb:
            active.append("ambiguous")
        if conf < threshold:
            active.append("low_confidence")
        if kind in {"fire", "rescue"}:
            active.append("protected_class")
        reasons.append("+".join(active) if active else "inside_delegation_boundary")
    return np.asarray(reasons, dtype=object)


def simulate_shift(
    benchmark: pd.DataFrame,
    target_mix: dict[str, float],
    regime_code: str,
    workload_multiplier: float,
    seed: int,
    params: HumanReviewParameters | None = None,
    explanation_code: str = "E2",
    ambiguity_multiplier: float = 1.0,
    high_consequence_multiplier: float = 1.0,
    target_ai_error_rate: float | None = None,
    selective_threshold: float = 0.80,
    governed_threshold: float = 0.72,
    governance_variant: str = "full",
    distribution_label: str = "CAT112_2025",
    return_event_log: bool = False,
) -> dict[str, float | int | str] | tuple[dict[str, float | int | str], pd.DataFrame]:
    params = params or HumanReviewParameters()
    explanation = EXPLANATIONS[explanation_code]
    regime = REGIMES[regime_code]
    rng = np.random.default_rng(seed)
    horizon = params.shift_hours * 3600.0
    expected_n = params.reference_arrivals_per_hour * workload_multiplier * params.shift_hours
    n = max(int(rng.poisson(expected_n)), 1)
    arrivals = np.sort(rng.uniform(0.0, horizon, size=n))
    cases = sample_cases(
        benchmark,
        n,
        rng,
        target_mix,
        ambiguity_multiplier=ambiguity_multiplier,
        high_consequence_multiplier=high_consequence_multiplier,
        target_ai_error_rate=target_ai_error_rate,
    )
    severity = cases["severity"].to_numpy(str)
    confidence = cases["ai_confidence_sim"].to_numpy(float)
    ambiguous = cases["ambiguous_sim"].to_numpy(bool)
    incident_type = cases["incident_type"].to_numpy(str)
    multi = cases["multi_service_predicted"].to_numpy(bool)
    ai_wrong = cases["ai_wrong_sim"].to_numpy(bool)
    ai_under = cases["ai_underdispatch_sim"].to_numpy(bool)
    ai_correct = ~ai_wrong

    formal = review_required(
        regime_code,
        severity,
        confidence,
        ambiguous,
        incident_type,
        selective_threshold=selective_threshold,
        governed_threshold=governed_threshold,
        governance_variant=governance_variant,
    )
    service_seconds = _review_times(rng, n, params)
    service_seconds += explanation["extra_seconds"] * params.explanation_time_multiplier
    service_seconds += ambiguous.astype(float) * 20.0
    service_seconds += (severity == "S2").astype(float) * 25.0
    service_seconds += multi.astype(float) * 15.0
    priority_score = (
        (severity == "S2").astype(float) * 100.0
        + np.isin(incident_type, ["fire", "rescue"]).astype(float) * 80.0
        + ambiguous.astype(float) * 60.0
        + (1.0 - confidence) * 40.0
    )
    if regime_code == "RP":
        starts, ends, server = _priority_schedule(
            arrivals, formal, service_seconds, params.effective_staff_count, priority_score
        )
        queue_policy = "risk_priority_nonpreemptive"
    else:
        starts, ends, server = _fifo_schedule(arrivals, formal, service_seconds, params.effective_staff_count)
        queue_policy = "fifo"

    decision_windows = np.array([DECISION_WINDOWS_SECONDS[value] for value in cases["reversibility"]])
    if regime.veto_window_seconds is not None:
        decision_windows = np.minimum(decision_windows, regime.veto_window_seconds)
    deadlines = np.minimum(arrivals + decision_windows, horizon)
    effective = formal & (ends <= deadlines)
    queue_delay = np.where(formal, starts - arrivals, 0.0)

    fatigue = 1.0 - params.fatigue_sensitivity * np.clip(queue_delay / 600.0, 0.0, 1.0)
    detection_probability = params.detection_probability * explanation["detection_multiplier"] * fatigue
    if regime.default_ai:
        detection_probability *= 1.0 - params.default_acceptance_effect
    detection_probability = np.clip(detection_probability, 0.0, 1.0)
    detected = ai_wrong & effective & (rng.random(n) < detection_probability)
    incorrect_override = ai_correct & effective & (rng.random(n) < params.incorrect_override_probability)

    if regime_code == "R0":
        ai_influences = np.zeros(n, dtype=bool)
        corrected = np.zeros(n, dtype=bool)
    else:
        ai_influences = np.ones(n, dtype=bool)
        corrected = detected
    delegated = ~formal
    no_modification_opportunity = delegated | (formal & ~effective)
    if regime_code == "R0":
        effective_ai_authority = np.zeros(n, dtype=bool)
    elif regime.default_ai or regime_code in {"R3", "R5"}:
        effective_ai_authority = no_modification_opportunity
    else:
        effective_ai_authority = formal & ~effective
    nominal_human = nominal_human_responsibility(regime_code, formal)
    authority_responsibility_mismatch = effective_ai_authority & nominal_human

    uaee = ai_wrong & ai_influences & ~corrected
    error_actionable = ai_wrong & formal & regime.recommendation_presented
    error_influential = ai_wrong & ai_influences
    error_nominal = ai_wrong & nominal_human & regime.recommendation_presented

    action_delay = np.zeros(n, dtype=float)
    if regime_code == "R4":
        action_delay = np.where(effective, np.minimum(ends - arrivals, regime.veto_window_seconds), regime.veto_window_seconds)
    elif regime_code in {"R2", "R3", "R5"}:
        action_delay[formal] = np.minimum(ends[formal], deadlines[formal]) - arrivals[formal]
    else:
        action_delay[formal] = ends[formal] - arrivals[formal]
    action_time = arrivals + action_delay

    busy_within_shift = np.zeros(n, dtype=float)
    for index in np.flatnonzero(formal):
        busy_within_shift[index] = max(0.0, min(ends[index], horizon) - min(max(starts[index], 0.0), horizon))
    available_review_seconds = params.effective_staff_count * horizon
    demanded_review_seconds = float(service_seconds[formal].sum())
    utilization = busy_within_shift.sum() / available_review_seconds
    backlog = int(np.sum(formal & (ends > horizon)))
    mean_queue_length, max_queue_length = _queue_statistics(arrivals, starts, formal, horizon)

    review_minutes = service_seconds * formal / 60.0
    total_review_minutes = review_minutes.sum()
    # HCHAC follows the study definition: all S2 cases plus institutionally
    # protected fire/rescue classes, regardless of whether a particular
    # authority rule also treats ambiguity as a mandatory escalation trigger.
    high_consequence = (severity == "S2") | np.isin(incident_type, ["fire", "rescue"])
    low_risk = severity == "S0"
    material_value = ai_wrong | high_consequence | ambiguous
    exclusive_error = ai_wrong
    exclusive_ambiguous = (~exclusive_error) & ambiguous
    exclusive_high_correct = (~exclusive_error) & (~exclusive_ambiguous) & high_consequence
    exclusive_low_other_correct = ~(exclusive_error | exclusive_ambiguous | exclusive_high_correct)
    mandatory = high_consequence | ambiguous | np.isin(incident_type, ["fire", "rescue"])
    traceability = np.ones(n, dtype=bool)
    independent_decision = effective & np.isin(regime_code, ["R0", "R1", "R3", "R5", "RP"])

    result: dict[str, float | int | str] = {
        "seed": seed,
        "regime": regime_code,
        "regime_name": regime.name,
        "workload_multiplier": workload_multiplier,
        "distribution": distribution_label,
        "governance_variant": governance_variant if regime_code == "R5" else "not_applicable",
        "queue_policy": queue_policy,
        "explanation": explanation_code,
        "n_incidents": n,
        "staff_count": params.effective_staff_count,
        "staffing_factor": params.staffing_factor,
        "review_time_scale": params.review_time_scale,
        "detection_probability_parameter": params.detection_probability,
        "default_acceptance_effect": params.default_acceptance_effect,
        "arrival_rate_per_hour": params.reference_arrivals_per_hour * workload_multiplier,
        "rho_h_required_over_available": demanded_review_seconds / available_review_seconds,
        "formal_human_oversight_rate": float(formal.mean()),
        "effective_human_review_rate": float(effective.mean()),
        "oversight_realization_gap": float(formal.mean() - effective.mean()),
        "effective_review_ratio": float(effective.sum() / formal.sum()) if formal.any() else float("nan"),
        "high_consequence_attention_coverage": float(effective[high_consequence].mean()) if high_consequence.any() else float("nan"),
        "successful_error_interception_rate": float(corrected.sum() / error_actionable.sum()) if error_actionable.any() else float("nan"),
        "error_interception_denominator": int(error_actionable.sum()),
        "uncorrected_ai_error_exposure": float(uaee.mean()),
        "uncorrected_ai_error_per_1000": float(uaee.mean() * 1000.0),
        "uncorrected_error_given_influential_ai_error": float(uaee.sum() / error_influential.sum()) if error_influential.any() else float("nan"),
        "failure_to_correct_conditional_ai_error": float(uaee[error_nominal].mean()) if error_nominal.any() else float("nan"),
        "reviews_attempted": int(formal.sum()),
        "reviews_started_within_shift": int(np.sum(formal & (starts <= horizon))),
        "reviews_completed_within_window": int(effective.sum()),
        "total_review_minutes_demanded": float(total_review_minutes),
        "completed_review_minutes": float((service_seconds * effective).sum() / 60.0),
        "completed_review_minutes_per_1000": float((service_seconds * effective).sum() / 60.0 / n * 1000.0),
        "review_minutes_per_1000": float(total_review_minutes / n * 1000.0),
        "available_review_minutes_per_shift": float(available_review_seconds / 60.0),
        "human_utilization": float(utilization),
        "review_backlog": backlog,
        "mean_queue_length": mean_queue_length,
        "max_queue_length": max_queue_length,
        "mean_queue_delay_seconds": _safe_mean(queue_delay, formal),
        "p95_queue_delay_seconds": float(np.nanpercentile(queue_delay[formal], 95)) if formal.any() else 0.0,
        "review_window_exceeded_rate": float(np.mean(formal & ~effective)),
        "mean_action_delay_seconds": float(action_delay.mean()),
        "median_action_delay_seconds": float(np.median(action_delay)),
        "p95_action_delay_seconds": float(np.percentile(action_delay, 95)),
        "decision_window_exceeded_rate": float(np.mean(action_delay > decision_windows)),
        "incident_throughput_per_hour": float(np.sum(action_time <= horizon) / params.shift_hours),
        "automatically_processed_rate": float(delegated.mean() if regime_code != "R4" else np.mean(~effective)),
        "mandatory_escalation_compliance": float(np.mean(formal[mandatory])) if mandatory.any() else float("nan"),
        "responsibility_traceability_rate": float(traceability.mean()),
        "independent_decision_rate": float(independent_decision.mean()),
        "effective_ai_authority_rate": float(effective_ai_authority.mean()),
        "nominal_human_responsibility_rate": float(nominal_human.mean()),
        "authority_responsibility_gap": float(authority_responsibility_mismatch.mean()),
        "ai_error_rate_simulated": float(ai_wrong.mean()),
        "ai_underdispatch_rate_simulated": float(ai_under.mean()),
        "reliability_perturbation_method": str(cases["reliability_perturbation_method"].iloc[0]),
        "incorrect_override_rate": float(incorrect_override.mean()),
        "mean_review_seconds_demanded": float(service_seconds[formal].mean()) if formal.any() else 0.0,
        "high_consequence_share": float(high_consequence.mean()),
        "ambiguity_share": float(ambiguous.mean()),
        "multi_service_predicted_share": float(multi.mean()),
        "attention_exclusive_ai_error_share": float(review_minutes[exclusive_error].sum() / total_review_minutes) if total_review_minutes else float("nan"),
        "attention_exclusive_ambiguous_share": float(review_minutes[exclusive_ambiguous].sum() / total_review_minutes) if total_review_minutes else float("nan"),
        "attention_exclusive_high_consequence_correct_share": float(review_minutes[exclusive_high_correct].sum() / total_review_minutes) if total_review_minutes else float("nan"),
        "attention_exclusive_low_other_correct_share": float(review_minutes[exclusive_low_other_correct].sum() / total_review_minutes) if total_review_minutes else float("nan"),
        "attention_allocation_efficiency": float(review_minutes[material_value].sum() / total_review_minutes) if total_review_minutes else float("nan"),
        "low_value_review_burden_minutes_per_1000": float(review_minutes[low_risk & ai_correct & ~ambiguous].sum() / n * 1000.0),
    }
    if not return_event_log:
        return result

    policy_reason = _policy_reason(severity, confidence, ambiguous, incident_type, governed_threshold)
    failure_mechanism = np.full(n, "none", dtype=object)
    failure_mechanism[uaee & delegated] = "delegated_ai_error"
    failure_mechanism[uaee & formal & ~effective] = "review_window_miss"
    failure_mechanism[uaee & effective & ~detected] = "reviewed_but_not_detected"
    failure_mechanism[(regime_code == "R4") & uaee & formal & ~effective] = "veto_window_expired"
    events = pd.DataFrame({
        "seed": seed,
        "regime": regime_code,
        "workload_multiplier": workload_multiplier,
        "incident_index": np.arange(n),
        "seed_incident_id": cases["seed_incident_id"].astype(str),
        "predispatch_text": cases["predispatch_text"].astype(str),
        "incident_type": incident_type,
        "severity": severity,
        "ambiguous": ambiguous,
        "ai_confidence": confidence,
        "ai_wrong": ai_wrong,
        "ai_underdispatch": ai_under,
        "formal_review_required": formal,
        "effective_review_completed": effective,
        "delegated": delegated,
        "corrected": corrected,
        "uaee": uaee,
        "high_consequence_review_miss": high_consequence & ~effective,
        "high_consequence_or_protected": high_consequence,
        "mandatory_case": mandatory,
        "policy_gate_reason": policy_reason,
        "failure_mechanism": failure_mechanism,
        "arrival_seconds": arrivals,
        "review_start_seconds": starts,
        "review_end_seconds": ends,
        "deadline_seconds": deadlines,
        "queue_delay_seconds": queue_delay,
        "service_seconds": service_seconds,
        "review_priority_score": priority_score,
        "server": server,
    })
    return result, events
