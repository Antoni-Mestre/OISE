from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import WORKLOAD_GRID


WORKLOADS = {f"W{index:02d}": value for index, value in enumerate(WORKLOAD_GRID)}


def cat112_target_mix(processed_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(processed_dir / "cat112_2025_incident_mix.csv")
    shares = dict(zip(frame["TIPUS"], frame["share"]))
    mapped = {
        "medical": shares.get("Assistència sanitària", 0.0),
        "security": shares.get("Seguretat", 0.0) + shares.get("Civisme", 0.0),
        "traffic_accident": shares.get("Trànsit", 0.0) + shares.get("Accident", 0.0),
        "fire": shares.get("Incendi", 0.0),
        "rescue": 0.0,
        "environment_weather_leak": shares.get("Fuita (aigua, gas, altres)", 0.0)
        + shares.get("Medi ambient", 0.0)
        + shares.get("Meteorologia", 0.0),
        "other": shares.get("Altres incidències", 0.0),
    }
    total = sum(mapped.values())
    return {key: value / total for key, value in mapped.items() if value > 0}


def cv112_target_mix(processed_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(processed_dir / "112cv_2025_incident_types.csv")
    shares = dict(zip(frame["incident_type_112cv"], frame["share"]))
    mapped = {
        "medical": shares.get("Sanitario", 0.0),
        "security": shares.get("Seguridad", 0.0),
        "traffic_accident": shares.get("Accidente", 0.0) + shares.get("Accidente Grave", 0.0),
        "fire": shares.get("Incendio", 0.0),
        "rescue": shares.get("Salvamento", 0.0),
        "environment_weather_leak": shares.get("Fenom. Natural", 0.0)
        + shares.get("Medioambiente", 0.0)
        + shares.get("Suministro Básico", 0.0),
        "other": sum(
            value
            for key, value in shares.items()
            if key
            not in {
                "Sanitario", "Seguridad", "Accidente", "Accidente Grave", "Incendio",
                "Salvamento", "Fenom. Natural", "Medioambiente", "Suministro Básico",
            }
        ),
    }
    total = sum(mapped.values())
    return {key: value / total for key, value in mapped.items() if value > 0}


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _set_reliability(
    cases: pd.DataFrame,
    rng: np.random.Generator,
    target_error_rate: float | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    wrong = cases["ai_wrong"].astype(bool).to_numpy().copy()
    under = cases["ai_underdispatch"].astype(bool).to_numpy().copy()
    if target_error_rate is None:
        return wrong, under, "frozen_baseline"
    target_error_rate = float(np.clip(target_error_rate, 0.0, 1.0))
    target_count = int(round(target_error_rate * len(cases)))
    current_count = int(wrong.sum())
    confidence = cases["ai_confidence"].to_numpy(float)
    if target_count > current_count:
        candidates = np.flatnonzero(~wrong)
        count = min(target_count - current_count, len(candidates))
        weights = np.clip(1.01 - confidence[candidates], 0.02, None)
        chosen = rng.choice(candidates, size=count, replace=False, p=weights / weights.sum())
        wrong[chosen] = True
        empirical_under_share = float(cases.loc[cases["ai_wrong"].astype(bool), "ai_underdispatch"].mean())
        empirical_under_share = empirical_under_share if np.isfinite(empirical_under_share) else 0.5
        under[chosen] = rng.random(count) < empirical_under_share
        method = "confidence_weighted_corruption"
    elif target_count < current_count:
        candidates = np.flatnonzero(wrong)
        count = min(current_count - target_count, len(candidates))
        weights = np.clip(confidence[candidates], 0.02, None)
        chosen = rng.choice(candidates, size=count, replace=False, p=weights / weights.sum())
        wrong[chosen] = False
        under[chosen] = False
        method = "confidence_weighted_error_repair"
    else:
        method = "frozen_baseline_exact_target"
    return wrong, under, method


def sample_cases(
    benchmark: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    target_mix: dict[str, float],
    ambiguity_multiplier: float = 1.0,
    high_consequence_multiplier: float = 1.0,
    target_ai_error_rate: float | None = None,
) -> pd.DataFrame:
    available_types = {label for label in target_mix if benchmark["incident_type"].eq(label).any()}
    if not available_types:
        raise ValueError("Target mix has no incident types represented in the frozen benchmark")
    types = np.array(sorted(available_types))
    probabilities = np.array([target_mix[label] for label in types], dtype=float)
    probabilities /= probabilities.sum()
    requested_types = rng.choice(types, size=n, p=probabilities)
    chosen = np.empty(n, dtype=int)
    for label in types:
        mask = requested_types == label
        pool = benchmark.index[benchmark["incident_type"].eq(label)].to_numpy()
        severity_weight = np.where(
            benchmark.loc[pool, "severity"].eq("S2").to_numpy(),
            high_consequence_multiplier,
            1.0,
        ).astype(float)
        severity_weight /= severity_weight.sum()
        chosen[mask] = rng.choice(pool, size=int(mask.sum()), replace=True, p=severity_weight)
    cases = benchmark.loc[chosen].reset_index(drop=True).copy()

    ambiguity = cases["ambiguous"].astype(bool).to_numpy().copy()
    if ambiguity_multiplier > 1:
        add_probability = min(1.0, (ambiguity_multiplier - 1.0) * max(ambiguity.mean(), 0.05))
        ambiguity |= (~ambiguity) & (rng.random(n) < add_probability)
    elif ambiguity_multiplier < 1:
        ambiguity &= rng.random(n) < ambiguity_multiplier
    cases["ambiguous_sim"] = ambiguity

    predicted_counts = cases["predicted_services"].map(lambda value: len(_json_list(value))).to_numpy()
    cases["multi_service_predicted"] = predicted_counts >= 2
    wrong, under, method = _set_reliability(cases, rng, target_ai_error_rate)
    cases["ai_wrong_sim"] = wrong
    cases["ai_underdispatch_sim"] = under
    cases["reliability_perturbation_method"] = method
    cases["ai_confidence_sim"] = cases["ai_confidence"].to_numpy(float)
    return cases
