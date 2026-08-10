from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import ttest_1samp

from authority_regimes import REGIMES
from config import (
    ABLATIONS,
    CONFIGS,
    EXECUTION_DATE,
    GLOBAL_SEED,
    LOGS,
    PREDICTIONS,
    PROCESSED,
    SENSITIVITY,
    SIM_AGG,
    SIM_RAW,
    STATISTICS,
    WORKLOAD_GRID,
)
from human_review_model import EXPLANATIONS, HumanReviewParameters
from institutional_model import simulate_shift
from workload_model import cat112_target_mix, cv112_target_mix


MAIN_REGIMES = ["R0", "R1", "R2", "R3", "R4", "R5"]
AI_REGIMES = ["R1", "R2", "R3", "R4", "R5"]
PRIMARY_METRICS = [
    "rho_h_required_over_available",
    "formal_human_oversight_rate",
    "effective_human_review_rate",
    "oversight_realization_gap",
    "effective_review_ratio",
    "high_consequence_attention_coverage",
    "successful_error_interception_rate",
    "uncorrected_ai_error_exposure",
    "uncorrected_ai_error_per_1000",
    "uncorrected_error_given_influential_ai_error",
    "review_minutes_per_1000",
    "completed_review_minutes_per_1000",
    "total_review_minutes_demanded",
    "completed_review_minutes",
    "available_review_minutes_per_shift",
    "human_utilization",
    "review_backlog",
    "mean_queue_length",
    "max_queue_length",
    "mean_queue_delay_seconds",
    "p95_queue_delay_seconds",
    "reviews_attempted",
    "reviews_completed_within_window",
    "review_window_exceeded_rate",
    "mean_action_delay_seconds",
    "median_action_delay_seconds",
    "p95_action_delay_seconds",
    "decision_window_exceeded_rate",
    "incident_throughput_per_hour",
    "automatically_processed_rate",
    "mandatory_escalation_compliance",
    "effective_ai_authority_rate",
    "nominal_human_responsibility_rate",
    "authority_responsibility_gap",
    "attention_exclusive_ai_error_share",
    "attention_exclusive_ambiguous_share",
    "attention_exclusive_high_consequence_correct_share",
    "attention_exclusive_low_other_correct_share",
    "attention_allocation_efficiency",
    "low_value_review_burden_minutes_per_1000",
    "ai_error_rate_simulated",
    "ai_underdispatch_rate_simulated",
]
CROSSOVER_METRICS = [
    "high_consequence_attention_coverage",
    "uncorrected_ai_error_exposure",
    "review_minutes_per_1000",
    "effective_human_review_rate",
]
CROSSOVER_PAIRS = [("R1", "R3"), ("R1", "R4"), ("R1", "R5"), ("R3", "R5")]
LOWER_IS_BETTER = {"uncorrected_ai_error_exposure", "review_minutes_per_1000"}


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(map(str, (GLOBAL_SEED, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def workload_code(value: float) -> str:
    return f"W{WORKLOAD_GRID.index(value):02d}"


def save_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, **kwargs)


def summarize(frame: pd.DataFrame, groups: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(groups, keys))
        row["replications"] = int(group["replication"].nunique()) if "replication" in group else len(group)
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            suffixes = ["mean", "median", "sd", "mcse", "ci_low", "ci_high", "p2_5", "p97_5"]
            if not len(values):
                row.update({f"{metric}_{suffix}": np.nan for suffix in suffixes})
                continue
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            mcse = sd / math.sqrt(len(values))
            row.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_sd": sd,
                    f"{metric}_mcse": mcse,
                    f"{metric}_ci_low": float(values.mean() - 1.96 * mcse),
                    f"{metric}_ci_high": float(values.mean() + 1.96 * mcse),
                    f"{metric}_p2_5": float(np.quantile(values, 0.025)),
                    f"{metric}_p97_5": float(np.quantile(values, 0.975)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_identifiers(
    result: dict[str, object], experiment: str, replication: int, **levels: object
) -> dict[str, object]:
    result["experiment"] = experiment
    result["replication"] = replication
    result.update(levels)
    return result


def run_main(
    benchmark: pd.DataFrame,
    target_mix: dict[str, float],
    replications: int,
    extension_replications: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for multiplier in WORKLOAD_GRID:
        code = workload_code(multiplier)
        for replication in range(replications):
            seed = deterministic_seed("main", code, replication)
            for regime in MAIN_REGIMES:
                result = simulate_shift(benchmark, target_mix, regime, multiplier, seed, explanation_code="E2")
                rows.append(add_identifiers(result, "main_workload", replication, workload=code))
    runs = pd.DataFrame(rows)
    convergence = convergence_statistics(runs)
    # Extend only unstable workload levels in staged checkpoints.  Extending
    # all regimes at a selected workload preserves paired comparisons.
    for target in [200, 400]:
        if target > extension_replications:
            continue
        unstable_workloads = sorted(
            convergence.loc[~convergence["stable"], "workload_multiplier"].unique()
        )
        additions: list[dict[str, object]] = []
        for multiplier in unstable_workloads:
            code = workload_code(float(multiplier))
            current = int(
                runs.loc[runs["workload_multiplier"].eq(multiplier), "replication"].max()
            ) + 1
            for replication in range(current, target):
                seed = deterministic_seed("main", code, replication)
                for regime in MAIN_REGIMES:
                    result = simulate_shift(
                        benchmark, target_mix, regime, float(multiplier), seed, explanation_code="E2"
                    )
                    additions.append(add_identifiers(result, "main_workload", replication, workload=code))
        if additions:
            runs = pd.concat([runs, pd.DataFrame(additions)], ignore_index=True)
            convergence = convergence_statistics(runs)
    summary = summarize(runs, ["regime", "regime_name", "workload", "workload_multiplier"], PRIMARY_METRICS)
    save_csv(runs, SIM_RAW / "main_workload_runs.csv")
    save_csv(summary, SIM_AGG / "main_workload_summary.csv")
    save_csv(convergence, STATISTICS / "monte_carlo_convergence.csv")
    return runs, summary, convergence


def convergence_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "effective_human_review_rate",
        "high_consequence_attention_coverage",
        "uncorrected_ai_error_exposure",
    ]
    rows: list[dict[str, object]] = []
    for (regime, workload, multiplier), group in runs.groupby(
        ["regime", "workload", "workload_multiplier"], sort=True
    ):
        group = group.sort_values("replication")
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            row: dict[str, object] = {
                "regime": regime,
                "workload": workload,
                "workload_multiplier": multiplier,
                "metric": metric,
                "available_replications": len(values),
            }
            for checkpoint in [25, 50, 100, 200, 400]:
                if len(values) >= checkpoint:
                    x = values[:checkpoint]
                    row[f"mean_{checkpoint}"] = float(x.mean())
                    row[f"mcse_{checkpoint}"] = float(x.std(ddof=1) / math.sqrt(checkpoint))
                else:
                    row[f"mean_{checkpoint}"] = np.nan
                    row[f"mcse_{checkpoint}"] = np.nan
            final_n = 400 if len(values) >= 400 else (200 if len(values) >= 200 else 100)
            prior_n = final_n // 2
            row["final_replications"] = final_n
            row["absolute_checkpoint_change"] = abs(row[f"mean_{final_n}"] - row[f"mean_{prior_n}"])
            row["stable"] = bool(
                row["absolute_checkpoint_change"] <= 0.015 and row[f"mcse_{final_n}"] <= 0.010
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_capacity_detection(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    staffing = [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
    detection = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    rows: list[dict[str, object]] = []
    for capacity in staffing:
        for probability in detection:
            params = HumanReviewParameters(staffing_factor=capacity, detection_probability=probability)
            for replication in range(replications):
                # Common random numbers across the factorial grid isolate the
                # parameter effects; replications remain independent.
                seed = deterministic_seed("capacity_detection", replication)
                for regime in ["R1", "R5"]:
                    result = simulate_shift(benchmark, target_mix, regime, 1.25, seed, params=params)
                    rows.append(
                        add_identifiers(
                            result,
                            "capacity_detection",
                            replication,
                            capacity_factor=capacity,
                            detection_probability=probability,
                        )
                    )
    runs = pd.DataFrame(rows)
    metrics = [
        "high_consequence_attention_coverage",
        "uncorrected_ai_error_exposure",
        "review_minutes_per_1000",
        "completed_review_minutes_per_1000",
        "available_review_minutes_per_shift",
        "effective_human_review_rate",
        "rho_h_required_over_available",
    ]
    summary = summarize(runs, ["regime", "capacity_factor", "detection_probability"], metrics)
    paired = paired_grid_differences(runs)
    decomposition = balanced_two_way_decomposition(paired)
    save_csv(runs, SENSITIVITY / "capacity_detection_runs.csv")
    save_csv(summary, SENSITIVITY / "capacity_detection_summary.csv")
    save_csv(paired, SENSITIVITY / "capacity_detection_paired_differences.csv")
    save_csv(decomposition, STATISTICS / "capacity_detection_variance_decomposition.csv")
    return runs, summary, paired, decomposition


def paired_grid_differences(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "high_consequence_attention_coverage",
        "uncorrected_ai_error_exposure",
        "review_minutes_per_1000",
        "completed_review_minutes_per_1000",
    ]
    index = ["capacity_factor", "detection_probability", "replication", "seed"]
    pieces = []
    for metric in metrics:
        wide = runs.pivot(index=index, columns="regime", values=metric).reset_index()
        wide["metric"] = metric
        wide["delta_R5_minus_R1"] = wide["R5"] - wide["R1"]
        pieces.append(wide[index + ["metric", "R1", "R5", "delta_R5_minus_R1"]])
    return pd.concat(pieces, ignore_index=True)


def balanced_two_way_decomposition(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, group in paired.groupby("metric"):
        y = group["delta_R5_minus_R1"].to_numpy(float)
        grand = float(y.mean())
        cell = group.groupby(["capacity_factor", "detection_probability"])["delta_R5_minus_R1"].mean()
        mean_capacity = group.groupby("capacity_factor")["delta_R5_minus_R1"].mean()
        mean_detection = group.groupby("detection_probability")["delta_R5_minus_R1"].mean()
        n_rep = int(group.groupby(["capacity_factor", "detection_probability"]).size().iloc[0])
        n_capacity = len(mean_capacity)
        n_detection = len(mean_detection)
        ss_capacity = n_detection * n_rep * float(((mean_capacity - grand) ** 2).sum())
        ss_detection = n_capacity * n_rep * float(((mean_detection - grand) ** 2).sum())
        interaction = 0.0
        for (capacity, detection), value in cell.items():
            interaction += (value - mean_capacity[capacity] - mean_detection[detection] + grand) ** 2
        ss_interaction = n_rep * float(interaction)
        fitted = group.apply(
            lambda r: cell[(r["capacity_factor"], r["detection_probability"])], axis=1
        ).to_numpy(float)
        ss_within = float(((y - fitted) ** 2).sum())
        ss_total = float(((y - grand) ** 2).sum())
        for component, value, df in [
            ("human_capacity", ss_capacity, n_capacity - 1),
            ("detection_probability", ss_detection, n_detection - 1),
            ("capacity_x_detection", ss_interaction, (n_capacity - 1) * (n_detection - 1)),
            ("within_cell_monte_carlo", ss_within, len(y) - n_capacity * n_detection),
        ]:
            rows.append(
                {
                    "metric": metric,
                    "component": component,
                    "sum_squares": value,
                    "degrees_freedom": df,
                    "variance_share": value / ss_total if ss_total else np.nan,
                    "balanced_replications_per_cell": n_rep,
                }
            )
    return pd.DataFrame(rows)


def run_review_time(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for scale in [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]:
        params = HumanReviewParameters(review_time_scale=scale)
        for multiplier in WORKLOAD_GRID:
            for replication in range(replications):
                seed = deterministic_seed("review_time", scale, workload_code(multiplier), replication)
                for regime in ["R1", "R3", "R5"]:
                    result = simulate_shift(benchmark, target_mix, regime, multiplier, seed, params=params)
                    rows.append(
                        add_identifiers(
                            result,
                            "review_time",
                            replication,
                            review_time_multiplier=scale,
                            workload=workload_code(multiplier),
                        )
                    )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs,
        ["regime", "review_time_multiplier", "workload", "workload_multiplier"],
        PRIMARY_METRICS,
    )
    save_csv(runs, SENSITIVITY / "review_time_runs.csv")
    save_csv(summary, SENSITIVITY / "review_time_summary.csv")
    return runs, summary


def run_ai_reliability(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios: dict[str, float | None] = {
        "lower_performing": 0.30,
        "baseline_frozen": None,
        "moderately_improved": 0.12,
        "high_performing": 0.05,
    }
    rows: list[dict[str, object]] = []
    for label, target in scenarios.items():
        for multiplier in WORKLOAD_GRID:
            for replication in range(replications):
                seed = deterministic_seed("ai_reliability", label, workload_code(multiplier), replication)
                for regime in ["R1", "R3", "R5"]:
                    result = simulate_shift(
                        benchmark,
                        target_mix,
                        regime,
                        multiplier,
                        seed,
                        target_ai_error_rate=target,
                    )
                    rows.append(
                        add_identifiers(
                            result,
                            "ai_reliability",
                            replication,
                            ai_reliability_scenario=label,
                            target_ai_error_rate="frozen" if target is None else target,
                            workload=workload_code(multiplier),
                        )
                    )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs,
        ["regime", "ai_reliability_scenario", "target_ai_error_rate", "workload", "workload_multiplier"],
        PRIMARY_METRICS,
    )
    save_csv(runs, SENSITIVITY / "ai_reliability_runs.csv")
    save_csv(summary, SENSITIVITY / "ai_reliability_summary.csv")
    return runs, summary


def run_ai_capacity_map(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios: dict[str, float | None] = {
        "lower_performing": 0.30,
        "baseline_frozen": None,
        "moderately_improved": 0.12,
        "high_performing": 0.05,
    }
    rows: list[dict[str, object]] = []
    for label, target in scenarios.items():
        for capacity in [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]:
            params = HumanReviewParameters(staffing_factor=capacity)
            for replication in range(replications):
                seed = deterministic_seed("ai_capacity", label, capacity, replication)
                for regime in ["R1", "R3", "R4", "R5"]:
                    result = simulate_shift(
                        benchmark,
                        target_mix,
                        regime,
                        1.25,
                        seed,
                        params=params,
                        target_ai_error_rate=target,
                    )
                    rows.append(
                        add_identifiers(
                            result,
                            "ai_capacity_map",
                            replication,
                            ai_reliability_scenario=label,
                            target_ai_error_rate="frozen" if target is None else target,
                            capacity_factor=capacity,
                        )
                    )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs,
        ["regime", "ai_reliability_scenario", "target_ai_error_rate", "capacity_factor"],
        PRIMARY_METRICS,
    )
    save_csv(runs, SENSITIVITY / "ai_capacity_map_runs.csv")
    save_csv(summary, SENSITIVITY / "ai_capacity_map_summary.csv")
    return runs, summary


def run_r5_ablations(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = ["confidence_only", "severity_only", "mandatory_only", "confidence_severity", "full"]
    selected_workloads = [0.75, 1.00, 1.25, 1.50, 2.00]
    rows: list[dict[str, object]] = []
    for variant in variants:
        for multiplier in selected_workloads:
            for replication in range(replications):
                seed = deterministic_seed("r5_ablation", variant, multiplier, replication)
                result = simulate_shift(
                    benchmark,
                    target_mix,
                    "R5",
                    multiplier,
                    seed,
                    governance_variant=variant,
                )
                rows.append(
                    add_identifiers(
                        result,
                        "r5_governance_ablation",
                        replication,
                        governance_variant=variant,
                        workload=workload_code(multiplier),
                    )
                )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs, ["governance_variant", "workload", "workload_multiplier"], PRIMARY_METRICS
    )
    save_csv(runs, ABLATIONS / "r5_governance_runs.csv")
    save_csv(summary, ABLATIONS / "r5_governance_summary.csv")
    return runs, summary


def run_default_effect(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for effect in [0.00, 0.10, 0.25, 0.50, 0.75]:
        params = HumanReviewParameters(default_acceptance_effect=effect)
        for multiplier in [0.75, 1.00, 1.25, 1.50, 2.00]:
            for replication in range(replications):
                seed = deterministic_seed("default_effect", effect, multiplier, replication)
                for regime in ["R2", "R4"]:
                    result = simulate_shift(benchmark, target_mix, regime, multiplier, seed, params=params)
                    rows.append(
                        add_identifiers(
                            result,
                            "default_effect_ablation",
                            replication,
                            default_acceptance_effect=effect,
                            workload=workload_code(multiplier),
                        )
                    )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs,
        ["regime", "default_acceptance_effect", "workload", "workload_multiplier"],
        PRIMARY_METRICS,
    )
    save_csv(runs, ABLATIONS / "default_effect_runs.csv")
    save_csv(summary, ABLATIONS / "default_effect_summary.csv")
    return runs, summary


def run_priority_control(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for multiplier in WORKLOAD_GRID:
        for replication in range(replications):
            seed = deterministic_seed("main", workload_code(multiplier), replication)
            result = simulate_shift(benchmark, target_mix, "RP", multiplier, seed)
            rows.append(
                add_identifiers(
                    result,
                    "universal_priority_control",
                    replication,
                    workload=workload_code(multiplier),
                )
            )
    runs = pd.DataFrame(rows)
    summary = summarize(runs, ["regime", "workload", "workload_multiplier"], PRIMARY_METRICS)
    save_csv(runs, ABLATIONS / "priority_control_runs.csv")
    save_csv(summary, ABLATIONS / "priority_control_summary.csv")
    return runs, summary


def run_distributional_sensitivity(
    benchmark: pd.DataFrame,
    mixes: dict[str, dict[str, float]],
    replications: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for label, target_mix in mixes.items():
        for multiplier in [0.75, 1.00, 1.25, 1.50, 2.00]:
            for replication in range(replications):
                seed = deterministic_seed("distribution", label, multiplier, replication)
                for regime in MAIN_REGIMES:
                    result = simulate_shift(
                        benchmark,
                        target_mix,
                        regime,
                        multiplier,
                        seed,
                        distribution_label=label,
                    )
                    rows.append(
                        add_identifiers(
                            result,
                            "distributional_sensitivity",
                            replication,
                            workload=workload_code(multiplier),
                        )
                    )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs, ["distribution", "regime", "workload", "workload_multiplier"], PRIMARY_METRICS
    )
    save_csv(runs, SENSITIVITY / "distributional_runs.csv")
    save_csv(summary, SENSITIVITY / "distributional_summary.csv")
    return runs, summary


def run_explanation_supplement(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for explanation in EXPLANATIONS:
        for multiplier in [1.00, 1.25, 1.50]:
            for replication in range(replications):
                seed = deterministic_seed("explanation", explanation, multiplier, replication)
                for regime in ["R1", "R5"]:
                    result = simulate_shift(
                        benchmark,
                        target_mix,
                        regime,
                        multiplier,
                        seed,
                        explanation_code=explanation,
                    )
                    rows.append(
                        add_identifiers(
                            result,
                            "explanation_supplement",
                            replication,
                            workload=workload_code(multiplier),
                        )
                    )
    runs = pd.DataFrame(rows)
    summary = summarize(
        runs, ["regime", "explanation", "workload", "workload_multiplier"], PRIMARY_METRICS
    )
    save_csv(runs, SENSITIVITY / "explanation_runs.csv")
    save_csv(summary, SENSITIVITY / "explanation_summary.csv")
    return runs, summary


def run_failure_logs(
    benchmark: pd.DataFrame, target_mix: dict[str, float], replications: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    for multiplier in [1.00, 1.25, 1.50, 2.00]:
        for replication in range(replications):
            seed = deterministic_seed("failure_log", multiplier, replication)
            for regime in ["R3", "R4", "R5"]:
                result, events = simulate_shift(
                    benchmark, target_mix, regime, multiplier, seed, return_event_log=True
                )
                run_rows.append(
                    add_identifiers(
                        result,
                        "failure_mode_logging",
                        replication,
                        workload=workload_code(multiplier),
                    )
                )
                events["replication"] = replication
                events["mandatory_exemption"] = events["mandatory_case"] & events["delegated"]
                events["policy_gate_failure"] = events["uaee"] & events["delegated"]
                event_frames.append(events)
    runs = pd.DataFrame(run_rows)
    events = pd.concat(event_frames, ignore_index=True)
    failures = events.loc[
        events["uaee"] | events["high_consequence_review_miss"] | events["mandatory_exemption"]
    ].copy()
    counts = (
        failures.groupby(
            ["regime", "workload_multiplier", "failure_mechanism"], dropna=False
        )
        .agg(
            event_count=("incident_index", "size"),
            uaee_count=("uaee", "sum"),
            high_consequence_review_miss_count=("high_consequence_review_miss", "sum"),
            mandatory_exemption_count=("mandatory_exemption", "sum"),
            unique_seed_incidents=("seed_incident_id", "nunique"),
        )
        .reset_index()
    )
    sample_parts = []
    eligible = failures.loc[failures["uaee"] | failures["mandatory_exemption"]].copy()
    for _, group in eligible.groupby(["regime", "failure_mechanism"], dropna=False):
        sample_parts.append(group.sample(min(5, len(group)), random_state=GLOBAL_SEED))
    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else eligible.head(0)
    save_csv(runs, SIM_RAW / "failure_mode_runs.csv")
    save_csv(events, SIM_RAW / "failure_event_log.csv.gz", compression="gzip")
    save_csv(counts, STATISTICS / "failure_mode_counts.csv")
    save_csv(sample, STATISTICS / "failure_mode_manual_inspection_sample.csv")
    return runs, counts, sample


def paired_effects(main_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(deterministic_seed("paired_effect_bootstrap"))
    for multiplier in [0.75, 1.00, 1.25, 1.50, 2.00]:
        subset = main_runs.loc[main_runs["workload_multiplier"].eq(multiplier)]
        for regime_a, regime_b in CROSSOVER_PAIRS:
            for metric in CROSSOVER_METRICS:
                wide = subset.pivot(index="replication", columns="regime", values=metric).dropna()
                diff = (wide[regime_a] - wide[regime_b]).to_numpy(float)
                boot = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
                test = ttest_1samp(diff, 0.0, nan_policy="omit")
                rows.append(
                    {
                        "workload_multiplier": multiplier,
                        "regime_a": regime_a,
                        "regime_b": regime_b,
                        "metric": metric,
                        "effect_a_minus_b": float(diff.mean()),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                        "paired_sd": float(diff.std(ddof=1)),
                        "standardized_paired_effect_dz": float(diff.mean() / diff.std(ddof=1))
                        if diff.std(ddof=1) > 0
                        else 0.0,
                        "raw_p_value": float(test.pvalue) if np.isfinite(test.pvalue) else 1.0,
                        "replications": len(diff),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm_adjust(frame["raw_p_value"].to_numpy(float))
    save_csv(frame, STATISTICS / "paired_regime_effects.csv")
    return frame


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _first_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    tolerance = 1e-12
    for index in range(len(x) - 1):
        y0, y1 = y[index], y[index + 1]
        if abs(y0) <= tolerance:
            continue
        if y0 * y1 < 0:
            return float(x[index] - y0 * (x[index + 1] - x[index]) / (y1 - y0))
    return None


def _favored(diff: float, metric: str, regime_a: str, regime_b: str) -> str:
    if abs(diff) < 1e-4:
        return "no_material_difference"
    a_better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
    return regime_a if a_better else regime_b


def crossover_analysis(main_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(deterministic_seed("crossover_bootstrap"))
    x = np.array(sorted(main_runs["workload_multiplier"].unique()), dtype=float)
    for regime_a, regime_b in CROSSOVER_PAIRS:
        for metric in CROSSOVER_METRICS:
            vectors: list[np.ndarray] = []
            means: list[float] = []
            ci_lows: list[float] = []
            ci_highs: list[float] = []
            rho_means: list[float] = []
            for multiplier in x:
                subset = main_runs.loc[main_runs["workload_multiplier"].eq(multiplier)]
                wide = subset.pivot(index="replication", columns="regime", values=metric).dropna()
                diff = (wide[regime_a] - wide[regime_b]).to_numpy(float)
                vectors.append(diff)
                means.append(float(diff.mean()))
                mcse = diff.std(ddof=1) / math.sqrt(len(diff))
                ci_lows.append(float(diff.mean() - 1.96 * mcse))
                ci_highs.append(float(diff.mean() + 1.96 * mcse))
                rho = subset.loc[subset["regime"].isin([regime_a, regime_b])].groupby("regime")[
                    "rho_h_required_over_available"
                ].mean()
                rho_means.append(float(rho.mean()))
            means_array = np.asarray(means)
            point = _first_crossing(x, means_array)
            bootstrap_points = []
            for _ in range(2000):
                boot_means = np.array(
                    [values[rng.integers(0, len(values), len(values))].mean() for values in vectors]
                )
                candidate = _first_crossing(x, boot_means)
                if candidate is not None:
                    bootstrap_points.append(candidate)
            if point is not None:
                rho_point = float(np.interp(point, x, np.asarray(rho_means)))
                status = "sign_crossover"
            else:
                rho_point = np.nan
                status = "no_sign_crossover_in_sweep"
            direction_positive = 1 if metric not in LOWER_IS_BETTER else -1
            onset_candidates = [
                value
                for value, low, high in zip(x, ci_lows, ci_highs)
                if (low > 0 and direction_positive == 1)
                or (high < 0 and direction_positive == -1)
            ]
            rows.append(
                {
                    "regime_a": regime_a,
                    "regime_b": regime_b,
                    "metric": metric,
                    "status": status,
                    "estimated_crossover_workload": point,
                    "crossover_workload_ci_low": float(np.quantile(bootstrap_points, 0.025))
                    if bootstrap_points
                    else np.nan,
                    "crossover_workload_ci_high": float(np.quantile(bootstrap_points, 0.975))
                    if bootstrap_points
                    else np.nan,
                    "bootstrap_crossing_probability": len(bootstrap_points) / 2000,
                    "estimated_pair_mean_rho_h": rho_point,
                    "practical_superiority_onset_for_a": min(onset_candidates)
                    if onset_candidates
                    else np.nan,
                    "regime_favored_at_lowest_workload": _favored(
                        means_array[0], metric, regime_a, regime_b
                    ),
                    "regime_favored_at_highest_workload": _favored(
                        means_array[-1], metric, regime_a, regime_b
                    ),
                    "difference_at_lowest_workload": means_array[0],
                    "difference_at_highest_workload": means_array[-1],
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, STATISTICS / "crossover_results.csv")
    return frame


def nondominated(points: pd.DataFrame) -> set[str]:
    efficient: set[str] = set()
    for _, candidate in points.iterrows():
        dominated = False
        for _, challenger in points.iterrows():
            if challenger["regime"] == candidate["regime"]:
                continue
            weak = (
                challenger["high_consequence_attention_coverage"]
                >= candidate["high_consequence_attention_coverage"]
                and challenger["uncorrected_ai_error_exposure"]
                <= candidate["uncorrected_ai_error_exposure"]
                and challenger["review_minutes_per_1000"] <= candidate["review_minutes_per_1000"]
            )
            strict = (
                challenger["high_consequence_attention_coverage"]
                > candidate["high_consequence_attention_coverage"] + 1e-12
                or challenger["uncorrected_ai_error_exposure"]
                < candidate["uncorrected_ai_error_exposure"] - 1e-12
                or challenger["review_minutes_per_1000"] < candidate["review_minutes_per_1000"] - 1e-12
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            efficient.add(str(candidate["regime"]))
    return efficient


def pareto_analysis(main_runs: pd.DataFrame, bootstrap_replications: int = 1000) -> pd.DataFrame:
    metrics = [
        "high_consequence_attention_coverage",
        "uncorrected_ai_error_exposure",
        "review_minutes_per_1000",
        "rho_h_required_over_available",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(deterministic_seed("pareto_bootstrap"))
    scoped = main_runs.loc[main_runs["regime"].isin(AI_REGIMES)].copy()
    for multiplier, group in scoped.groupby("workload_multiplier", sort=True):
        means = group.groupby("regime")[metrics].mean().reset_index()
        efficient = nondominated(means)
        counts = {regime: 0 for regime in AI_REGIMES}
        # Vectorised paired bootstrap.  The array is replication x regime x
        # objective, so every resample preserves the common incident-stream
        # draw across regimes without repeatedly concatenating DataFrames.
        regimes = sorted(group["regime"].unique())
        replications = sorted(group["replication"].unique())
        arrays = []
        for metric in metrics[:3]:
            matrix = (
                group.pivot(index="replication", columns="regime", values=metric)
                .reindex(index=replications, columns=regimes)
                .to_numpy(float)
            )
            arrays.append(matrix)
        objective_array = np.stack(arrays, axis=2)
        sampled_indices = rng.integers(
            0, len(replications), size=(bootstrap_replications, len(replications))
        )
        boot_means = objective_array[sampled_indices].mean(axis=1)
        for boot in boot_means:
            for candidate_index, regime in enumerate(regimes):
                candidate = boot[candidate_index]
                dominated = False
                for challenger_index in range(len(regimes)):
                    if challenger_index == candidate_index:
                        continue
                    challenger = boot[challenger_index]
                    weak = (
                        challenger[0] >= candidate[0]
                        and challenger[1] <= candidate[1]
                        and challenger[2] <= candidate[2]
                    )
                    strict = (
                        challenger[0] > candidate[0] + 1e-12
                        or challenger[1] < candidate[1] - 1e-12
                        or challenger[2] < candidate[2] - 1e-12
                    )
                    if weak and strict:
                        dominated = True
                        break
                if not dominated:
                    counts[regime] += 1
        for _, point in means.iterrows():
            rows.append(
                {
                    "workload_multiplier": multiplier,
                    "regime": point["regime"],
                    "hchac_mean": point["high_consequence_attention_coverage"],
                    "uaee_mean": point["uncorrected_ai_error_exposure"],
                    "review_minutes_per_1000_mean": point["review_minutes_per_1000"],
                    "rho_h_mean": point["rho_h_required_over_available"],
                    "pareto_efficient_point_estimate": point["regime"] in efficient,
                    "bootstrap_pareto_probability": counts[point["regime"]] / bootstrap_replications,
                    "scope_note": "R1-R5 only; R0 has no AI recommendation pathway and structurally zero UAEE",
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, STATISTICS / "pareto_frontier.csv")
    return frame


def capacity_matched_summary(capacity_summary: pd.DataFrame) -> pd.DataFrame:
    frame = capacity_summary.loc[capacity_summary["detection_probability"].eq(0.85)].copy()
    keep = [
        "regime",
        "capacity_factor",
        "replications",
        "available_review_minutes_per_shift_mean",
        "rho_h_required_over_available_mean",
        "high_consequence_attention_coverage_mean",
        "high_consequence_attention_coverage_ci_low",
        "high_consequence_attention_coverage_ci_high",
        "completed_review_minutes_per_1000_mean",
        "review_minutes_per_1000_mean",
        "uncorrected_ai_error_exposure_mean",
    ]
    frame = frame[keep]
    save_csv(frame, STATISTICS / "capacity_matched_comparisons.csv")
    target_rows: list[dict[str, object]] = []
    for regime, group in frame.groupby("regime"):
        group = group.sort_values("available_review_minutes_per_shift_mean")
        for target in [0.80, 0.90, 0.95]:
            achieved = group.loc[group["high_consequence_attention_coverage_ci_low"] >= target]
            if len(achieved):
                row = achieved.iloc[0]
                target_rows.append(
                    {
                        "regime": regime,
                        "hchac_target": target,
                        "minimum_tested_capacity_factor": row["capacity_factor"],
                        "available_review_minutes_per_shift": row[
                            "available_review_minutes_per_shift_mean"
                        ],
                        "completed_review_minutes_per_1000": row[
                            "completed_review_minutes_per_1000_mean"
                        ],
                        "criterion": "lower 95% Monte Carlo CI bound reaches target",
                    }
                )
            else:
                target_rows.append(
                    {
                        "regime": regime,
                        "hchac_target": target,
                        "minimum_tested_capacity_factor": np.nan,
                        "available_review_minutes_per_shift": np.nan,
                        "completed_review_minutes_per_1000": np.nan,
                        "criterion": "target not reached within tested capacity grid",
                    }
                )
    targets = pd.DataFrame(target_rows)
    save_csv(targets, STATISTICS / "human_minutes_for_hchac_targets.csv")
    return targets


def transition_analysis(main_summary: pd.DataFrame) -> dict[str, object]:
    r1 = main_summary.loc[main_summary["regime"].eq("R1")].sort_values("workload_multiplier")
    gap_transition = r1.loc[r1["oversight_realization_gap_mean"] >= 0.10]
    ratio_transition = r1.loc[r1["effective_review_ratio_mean"] < 0.90]
    result = {
        "operational_definition_rho_h": "demanded human review seconds / (integer effective reviewers * shift seconds)",
        "capacity_abundant_descriptive_rule": "rho_H < 0.8",
        "transition_descriptive_rule": "0.8 <= rho_H <= 1.2",
        "overloaded_descriptive_rule": "rho_H > 1.2",
        "first_workload_R1_ORG_at_least_0_10": float(gap_transition.iloc[0]["workload_multiplier"])
        if len(gap_transition)
        else None,
        "rho_h_at_first_R1_ORG_at_least_0_10": float(
            gap_transition.iloc[0]["rho_h_required_over_available_mean"]
        )
        if len(gap_transition)
        else None,
        "first_workload_R1_effective_review_ratio_below_0_90": float(
            ratio_transition.iloc[0]["workload_multiplier"]
        )
        if len(ratio_transition)
        else None,
        "rho_h_at_first_R1_effective_review_ratio_below_0_90": float(
            ratio_transition.iloc[0]["rho_h_required_over_available_mean"]
        )
        if len(ratio_transition)
        else None,
    }
    (STATISTICS / "capacity_transition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_configuration(args: argparse.Namespace, benchmark: pd.DataFrame) -> None:
    config = {
        "execution_date": EXECUTION_DATE,
        "global_seed": GLOBAL_SEED,
        "main_regimes": MAIN_REGIMES,
        "priority_control_regime": "RP",
        "workload_grid": WORKLOAD_GRID,
        "main_replications": args.main_replications,
        "main_extension_replications": args.extension_replications,
        "sensitivity_replications": args.sensitivity_replications,
        "primary_robustness_replications": args.robustness_replications,
        "ablation_replications": args.ablation_replications,
        "failure_log_replications": args.failure_replications,
        "human_review_parameters": asdict(HumanReviewParameters()),
        "explanation_default": "E2",
        "r3_implementation_preserved": "delegates only S0, high-confidence, unambiguous cases",
        "hchac_definition": "S2 or protected fire/rescue case completed within decision/review window",
        "rho_h_definition": "demanded human review seconds divided by effective reviewer count times shift seconds",
        "frozen_benchmark_rows": len(benchmark),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (CONFIGS / "simulation_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised capacity-aware OISE experiments")
    parser.add_argument("--main-replications", type=int, default=100)
    parser.add_argument("--extension-replications", type=int, default=400)
    parser.add_argument("--sensitivity-replications", type=int, default=60)
    parser.add_argument("--robustness-replications", type=int, default=100)
    parser.add_argument("--ablation-replications", type=int, default=100)
    parser.add_argument("--failure-replications", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Smoke-test every branch with small replication counts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.main_replications = 10
        args.extension_replications = 10
        args.sensitivity_replications = 2
        args.robustness_replications = 2
        args.ablation_replications = 2
        args.failure_replications = 1
    started = time.time()
    benchmark = pd.read_csv(PREDICTIONS / "frozen_ai_predictions.csv")
    cat_mix = cat112_target_mix(PROCESSED)
    cv_mix = cv112_target_mix(PROCESSED)
    write_configuration(args, benchmark)
    stages: list[dict[str, object]] = []

    def stage(name: str, function, *function_args):
        before = time.time()
        print(f"START {name}", flush=True)
        output = function(*function_args)
        rows = len(output[0]) if isinstance(output, tuple) and hasattr(output[0], "__len__") else None
        stages.append({"stage": name, "seconds": time.time() - before, "primary_rows": rows})
        print(f"DONE {name} ({time.time() - before:.1f}s)", flush=True)
        return output

    main_runs, main_summary, convergence = stage(
        "main_workload",
        run_main,
        benchmark,
        cat_mix,
        args.main_replications,
        args.extension_replications,
    )
    capacity_runs, capacity_summary, paired_grid, decomposition = stage(
        "capacity_detection",
        run_capacity_detection,
        benchmark,
        cat_mix,
        args.robustness_replications,
    )
    review_runs, review_summary = stage(
        "review_time", run_review_time, benchmark, cat_mix, args.sensitivity_replications
    )
    ai_runs, ai_summary = stage(
        "ai_reliability", run_ai_reliability, benchmark, cat_mix, args.sensitivity_replications
    )
    ai_capacity_runs, ai_capacity_summary = stage(
        "ai_capacity_map", run_ai_capacity_map, benchmark, cat_mix, args.sensitivity_replications
    )
    r5_runs, r5_summary = stage(
        "r5_ablations", run_r5_ablations, benchmark, cat_mix, args.ablation_replications
    )
    default_runs, default_summary = stage(
        "default_effect", run_default_effect, benchmark, cat_mix, args.ablation_replications
    )
    priority_runs, priority_summary = stage(
        "priority_control", run_priority_control, benchmark, cat_mix, args.main_replications
    )
    distribution_runs, distribution_summary = stage(
        "distributional_sensitivity",
        run_distributional_sensitivity,
        benchmark,
        {"CAT112_2025": cat_mix, "112CV_2025": cv_mix},
        args.ablation_replications,
    )
    explanation_runs, explanation_summary = stage(
        "explanation_supplement",
        run_explanation_supplement,
        benchmark,
        cat_mix,
        args.sensitivity_replications,
    )
    failure_runs, failure_counts, failure_sample = stage(
        "failure_logs", run_failure_logs, benchmark, cat_mix, args.failure_replications
    )
    effects = paired_effects(main_runs)
    crossovers = crossover_analysis(main_runs)
    pareto = pareto_analysis(main_runs, bootstrap_replications=200 if args.quick else 1000)
    targets = capacity_matched_summary(capacity_summary)
    transition = transition_analysis(main_summary)

    run_frames = [
        main_runs,
        capacity_runs,
        review_runs,
        ai_runs,
        ai_capacity_runs,
        r5_runs,
        default_runs,
        priority_runs,
        distribution_runs,
        explanation_runs,
        failure_runs,
    ]
    status = {
        "quick_mode": args.quick,
        "simulation_calls": int(sum(len(frame) for frame in run_frames)),
        "main_run_rows": len(main_runs),
        "main_regime_workload_cells": int(main_runs.groupby(["regime", "workload_multiplier"]).ngroups),
        "main_convergence_pass_rate": float(convergence["stable"].mean()),
        "main_cells_extended_to_200": int((convergence["final_replications"] == 200).sum()),
        "capacity_detection_run_rows": len(capacity_runs),
        "failure_event_log_rows": int(
            sum(1 for _ in pd.read_csv(SIM_RAW / "failure_event_log.csv.gz", chunksize=100000))
        ),
        "crossover_rows": len(crossovers),
        "pareto_rows": len(pareto),
        "paired_effect_rows": len(effects),
        "wall_seconds": time.time() - started,
        "stages": stages,
        "transition": transition,
    }
    # Correct the chunk-count placeholder with the actual compressed log row count.
    status["failure_event_log_rows"] = int(
        sum(len(chunk) for chunk in pd.read_csv(SIM_RAW / "failure_event_log.csv.gz", chunksize=100000))
    )
    (LOGS / "simulation_execution.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (LOGS / "exact_commands.txt").write_text(
        "python scripts/prepare_revised_data.py\npython scripts/run_revised_experiments.py\n"
        "python scripts/make_revised_outputs.py\npython scripts/verify_revised_package.py\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
