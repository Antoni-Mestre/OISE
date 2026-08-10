from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import ABLATIONS, GLOBAL_SEED, SENSITIVITY, SIM_AGG, SIM_RAW, STATISTICS
from run_revised_experiments import deterministic_seed, nondominated, save_csv


PAIRS = [("R1", "R3"), ("R1", "R4"), ("R1", "R5"), ("R3", "R5")]
METRICS = [
    "high_consequence_attention_coverage",
    "uncorrected_ai_error_exposure",
    "review_minutes_per_1000",
    "effective_human_review_rate",
]
LOWER_IS_BETTER = {"uncorrected_ai_error_exposure", "review_minutes_per_1000"}


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    if not len(values):
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def all_crossings(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float = 1e-10,
    practical_epsilon: float = 0.0,
) -> list[float]:
    points: list[float] = []
    for index in range(len(x) - 1):
        y0, y1 = float(y[index]), float(y[index + 1])
        if abs(y0) <= tolerance and abs(y1) <= tolerance:
            continue
        if abs(y0) <= tolerance:
            points.append(float(x[index]))
        elif y0 * y1 < 0 and min(abs(y0), abs(y1)) >= practical_epsilon:
            points.append(float(x[index] - y0 * (x[index + 1] - x[index]) / (y1 - y0)))
    deduplicated: list[float] = []
    for point in points:
        if not deduplicated or abs(point - deduplicated[-1]) > 1e-6:
            deduplicated.append(point)
    return deduplicated


def favored(diff: float, metric: str, a: str, b: str) -> str:
    if abs(diff) < 1e-4:
        return "no_material_difference"
    a_is_better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
    return a if a_is_better else b


def detailed_crossovers(
    runs: pd.DataFrame,
    group_columns: list[str],
    pairs: list[tuple[str, str]],
    metrics: list[str],
    output: Path,
    bootstrap_replications: int = 1000,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(deterministic_seed("detailed_crossovers", output.name))
    grouped = [((), runs)] if not group_columns else list(runs.groupby(group_columns, dropna=False, sort=True))
    for group_keys, group in grouped:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        group_values = dict(zip(group_columns, group_keys))
        x = np.array(sorted(group["workload_multiplier"].unique()), dtype=float)
        for a, b in pairs:
            for metric in metrics:
                practical_epsilon = 1.0 if metric == "review_minutes_per_1000" else 0.0005
                vectors: list[np.ndarray] = []
                means: list[float] = []
                rho: list[float] = []
                for multiplier in x:
                    cell = group.loc[group["workload_multiplier"].eq(multiplier)]
                    index = ["replication"]
                    wide = cell.pivot_table(index=index, columns="regime", values=metric, aggfunc="first")
                    if a not in wide or b not in wide:
                        vectors = []
                        break
                    diff = (wide[a] - wide[b]).dropna().to_numpy(float)
                    vectors.append(diff)
                    means.append(float(diff.mean()))
                    rho_cell = cell.loc[cell["regime"].isin([a, b])].groupby("regime")[
                        "rho_h_required_over_available"
                    ].mean()
                    rho.append(float(rho_cell.mean()))
                if not vectors:
                    continue
                mean_array = np.asarray(means)
                points = all_crossings(x, mean_array, practical_epsilon=practical_epsilon)
                bootstrap_points: list[list[float]] = []
                for _ in range(bootstrap_replications):
                    boot_means = np.array(
                        [values[rng.integers(0, len(values), len(values))].mean() for values in vectors]
                    )
                    bootstrap_points.append(
                        all_crossings(x, boot_means, practical_epsilon=practical_epsilon)
                    )
                if not points:
                    rows.append(
                        {
                            **group_values,
                            "regime_a": a,
                            "regime_b": b,
                            "metric": metric,
                            "crossing_index": 0,
                            "status": "no_sign_crossover_in_sweep",
                            "estimated_crossover_workload": np.nan,
                            "crossover_workload_ci_low": np.nan,
                            "crossover_workload_ci_high": np.nan,
                            "bootstrap_crossing_probability": float(
                                np.mean([len(value) > 0 for value in bootstrap_points])
                            ),
                            "estimated_pair_mean_rho_h": np.nan,
                            "favored_at_lowest_workload": favored(mean_array[0], metric, a, b),
                            "favored_at_highest_workload": favored(mean_array[-1], metric, a, b),
                            "difference_at_lowest_workload": mean_array[0],
                            "difference_at_highest_workload": mean_array[-1],
                        }
                    )
                    continue
                for crossing_index, point in enumerate(points, start=1):
                    segment_index = max(
                        0,
                        min(len(x) - 2, int(np.searchsorted(x, point, side="right") - 1)),
                    )
                    candidates = [
                        value[crossing_index - 1]
                        for value in bootstrap_points
                        if len(value) >= crossing_index
                    ]
                    rows.append(
                        {
                            **group_values,
                            "regime_a": a,
                            "regime_b": b,
                            "metric": metric,
                            "crossing_index": crossing_index,
                            "status": "sign_crossover",
                            "estimated_crossover_workload": point,
                            "crossover_workload_ci_low": float(np.quantile(candidates, 0.025))
                            if candidates
                            else np.nan,
                            "crossover_workload_ci_high": float(np.quantile(candidates, 0.975))
                            if candidates
                            else np.nan,
                            "bootstrap_crossing_probability": len(candidates) / bootstrap_replications,
                            "estimated_pair_mean_rho_h": float(np.interp(point, x, np.asarray(rho))),
                            "favored_below_crossing": favored(
                                mean_array[segment_index], metric, a, b
                            ),
                            "favored_above_crossing": favored(
                                mean_array[segment_index + 1], metric, a, b
                            ),
                            "favored_at_lowest_workload": favored(mean_array[0], metric, a, b),
                            "favored_at_highest_workload": favored(mean_array[-1], metric, a, b),
                            "difference_at_lowest_workload": mean_array[0],
                            "difference_at_highest_workload": mean_array[-1],
                        }
                    )
    frame = pd.DataFrame(rows)
    save_csv(frame, output)
    return frame


def priority_effects(main_runs: pd.DataFrame, priority_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "high_consequence_attention_coverage",
        "uncorrected_ai_error_exposure",
        "review_minutes_per_1000",
        "completed_review_minutes_per_1000",
        "effective_human_review_rate",
        "mean_action_delay_seconds",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(deterministic_seed("priority_paired_effects"))
    main = main_runs.loc[main_runs["replication"].lt(100) & main_runs["regime"].isin(["R1", "R3", "R5"])]
    combined = pd.concat([main, priority_runs], ignore_index=True)
    for multiplier, cell in combined.groupby("workload_multiplier", sort=True):
        for comparator in ["R1", "R3", "R5"]:
            for metric in metrics:
                wide = cell.pivot_table(index="replication", columns="regime", values=metric, aggfunc="first")
                diff = (wide["RP"] - wide[comparator]).dropna().to_numpy(float)
                low, high = bootstrap_mean_ci(diff, rng)
                rows.append(
                    {
                        "workload_multiplier": multiplier,
                        "comparison": f"RP_minus_{comparator}",
                        "metric": metric,
                        "paired_mean_difference": float(diff.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "paired_sd": float(diff.std(ddof=1)),
                        "replications": len(diff),
                    }
                )
    frame = pd.DataFrame(rows)
    save_csv(frame, STATISTICS / "priority_control_paired_effects.csv")
    return frame


def delta_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(SENSITIVITY / "capacity_detection_paired_differences.csv")
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(["capacity_factor", "detection_probability", "metric"], sort=True):
        values = group["delta_R5_minus_R1"].to_numpy(float)
        mcse = float(values.std(ddof=1) / math.sqrt(len(values)))
        rows.append(
            {
                "capacity_factor": keys[0],
                "detection_probability": keys[1],
                "metric": keys[2],
                "delta_R5_minus_R1_mean": float(values.mean()),
                "delta_ci_low": float(values.mean() - 1.96 * mcse),
                "delta_ci_high": float(values.mean() + 1.96 * mcse),
                "delta_mcse": mcse,
                "replications": len(values),
            }
        )
    capacity_delta = pd.DataFrame(rows)
    save_csv(capacity_delta, SENSITIVITY / "capacity_detection_delta_summary.csv")

    runs = pd.read_csv(SENSITIVITY / "ai_capacity_map_runs.csv", low_memory=False)
    metrics = ["high_consequence_attention_coverage", "uncorrected_ai_error_exposure", "review_minutes_per_1000"]
    ai_rows: list[dict[str, object]] = []
    for (scenario, capacity), group in runs.groupby(["ai_reliability_scenario", "capacity_factor"], sort=True):
        for metric in metrics:
            wide = group.pivot_table(index="replication", columns="regime", values=metric, aggfunc="first")
            diff = (wide["R5"] - wide["R1"]).dropna().to_numpy(float)
            mcse = float(diff.std(ddof=1) / math.sqrt(len(diff)))
            ai_rows.append(
                {
                    "ai_reliability_scenario": scenario,
                    "capacity_factor": capacity,
                    "metric": metric,
                    "delta_R5_minus_R1_mean": float(diff.mean()),
                    "delta_ci_low": float(diff.mean() - 1.96 * mcse),
                    "delta_ci_high": float(diff.mean() + 1.96 * mcse),
                    "replications": len(diff),
                    "achieved_ai_error_rate": float(group["ai_error_rate_simulated"].mean()),
                }
            )
    ai_delta = pd.DataFrame(ai_rows)
    save_csv(ai_delta, SENSITIVITY / "ai_capacity_delta_summary.csv")
    return capacity_delta, ai_delta


def ai_capacity_pareto() -> pd.DataFrame:
    runs = pd.read_csv(SENSITIVITY / "ai_capacity_map_runs.csv", low_memory=False)
    rows: list[dict[str, object]] = []
    for (scenario, capacity), group in runs.groupby(["ai_reliability_scenario", "capacity_factor"], sort=True):
        points = (
            group.groupby("regime")[[
                "high_consequence_attention_coverage",
                "uncorrected_ai_error_exposure",
                "review_minutes_per_1000",
            ]]
            .mean()
            .reset_index()
        )
        frontier = nondominated(points)
        for _, point in points.iterrows():
            rows.append(
                {
                    "ai_reliability_scenario": scenario,
                    "achieved_ai_error_rate": float(group["ai_error_rate_simulated"].mean()),
                    "capacity_factor": capacity,
                    "regime": point["regime"],
                    "hchac_mean": point["high_consequence_attention_coverage"],
                    "uaee_mean": point["uncorrected_ai_error_exposure"],
                    "review_minutes_per_1000_mean": point["review_minutes_per_1000"],
                    "pareto_efficient": point["regime"] in frontier,
                    "frontier_regimes": "+".join(sorted(frontier)),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, STATISTICS / "ai_capacity_pareto_map.csv")
    return frame


def interpolate_threshold(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    for index in range(len(x) - 1):
        y0, y1 = y[index], y[index + 1]
        if (y0 - threshold) * (y1 - threshold) <= 0 and y0 != y1:
            return float(x[index] + (threshold - y0) * (x[index + 1] - x[index]) / (y1 - y0))
    return None


def transition_details(main_runs: pd.DataFrame, main_summary: pd.DataFrame) -> dict[str, object]:
    r1_summary = main_summary.loc[main_summary["regime"].eq("R1")].sort_values("workload_multiplier")
    x = r1_summary["workload_multiplier"].to_numpy(float)
    rho = r1_summary["rho_h_required_over_available_mean"].to_numpy(float)
    ehrr = r1_summary["effective_human_review_rate_mean"].to_numpy(float)
    slopes = np.diff(ehrr) / np.diff(rho)
    steepest = int(np.argmin(slopes))
    majority_workload = interpolate_threshold(x, ehrr, 0.50)
    majority_rho = float(np.interp(majority_workload, x, rho)) if majority_workload is not None else None
    rho_one_workload = interpolate_threshold(x, rho, 1.0)

    rng = np.random.default_rng(deterministic_seed("transition_bootstrap"))
    r1_runs = main_runs.loc[main_runs["regime"].eq("R1")]
    boot_majority: list[float] = []
    boot_steep_mid_rho: list[float] = []
    for _ in range(2000):
        means = []
        rhos = []
        for multiplier in x:
            group = r1_runs.loc[r1_runs["workload_multiplier"].eq(multiplier)]
            indices = rng.integers(0, len(group), len(group))
            sample = group.iloc[indices]
            means.append(float(sample["effective_human_review_rate"].mean()))
            rhos.append(float(sample["rho_h_required_over_available"].mean()))
        means_array = np.asarray(means)
        rhos_array = np.asarray(rhos)
        candidate = interpolate_threshold(x, means_array, 0.50)
        if candidate is not None:
            boot_majority.append(candidate)
        boot_slopes = np.diff(means_array) / np.diff(rhos_array)
        index = int(np.argmin(boot_slopes))
        boot_steep_mid_rho.append(float((rhos_array[index] + rhos_array[index + 1]) / 2))
    result = {
        "rho_h_definition": "demanded review seconds / (integer effective reviewers * shift seconds)",
        "deadline_limited_at_minimum_tested_load": bool(ehrr[0] < 0.95),
        "minimum_tested_workload": float(x[0]),
        "rho_h_at_minimum_tested_workload": float(rho[0]),
        "R1_EHRR_at_minimum_tested_workload": float(ehrr[0]),
        "steepest_decline_workload_interval": [float(x[steepest]), float(x[steepest + 1])],
        "steepest_decline_rho_h_interval": [float(rho[steepest]), float(rho[steepest + 1])],
        "steepest_decline_slope_dEHRR_per_rho_h": float(slopes[steepest]),
        "steepest_decline_midpoint_rho_h_bootstrap_ci": [
            float(np.quantile(boot_steep_mid_rho, 0.025)),
            float(np.quantile(boot_steep_mid_rho, 0.975)),
        ],
        "estimated_workload_where_R1_EHRR_falls_below_0_50": majority_workload,
        "estimated_workload_where_R1_EHRR_0_50_ci": [
            float(np.quantile(boot_majority, 0.025)),
            float(np.quantile(boot_majority, 0.975)),
        ]
        if boot_majority
        else [None, None],
        "estimated_rho_h_where_R1_EHRR_falls_below_0_50": majority_rho,
        "estimated_workload_where_R1_rho_h_equals_1": rho_one_workload,
        "interpretation": "Deadline misses begin below rho_H=1; the capacity-driven cliff is the steepest decline interval, not the first nonzero ORG.",
    }
    (STATISTICS / "capacity_transition_detailed.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def failure_taxonomy() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(SIM_RAW / "failure_event_log.csv.gz", low_memory=False)
    conditions = [
        events["mandatory_exemption"].astype(bool),
        events["uaee"].astype(bool) & events["delegated"].astype(bool),
        events["uaee"].astype(bool) & events["formal_review_required"].astype(bool) & ~events["effective_review_completed"].astype(bool),
        events["uaee"].astype(bool) & events["effective_review_completed"].astype(bool) & ~events["corrected"].astype(bool),
        events["high_consequence_review_miss"].astype(bool) & ~events["uaee"].astype(bool),
    ]
    choices = [
        "mandatory_case_incorrectly_exempted",
        "delegated_ai_error",
        "review_or_veto_window_miss",
        "reviewed_but_error_not_detected",
        "high_consequence_review_miss_without_ai_error",
    ]
    events["exclusive_failure_taxon"] = np.select(conditions, choices, default="no_counted_failure")
    failure = events.loc[events["exclusive_failure_taxon"].ne("no_counted_failure")]
    counts = (
        failure.groupby(["regime", "workload_multiplier", "exclusive_failure_taxon"])
        .agg(
            event_count=("incident_index", "size"),
            unique_seed_incidents=("seed_incident_id", "nunique"),
            replications=("replication", "nunique"),
        )
        .reset_index()
    )
    denominators = events.groupby(["regime", "workload_multiplier"]).size().rename("all_events").reset_index()
    counts = counts.merge(denominators, on=["regime", "workload_multiplier"], how="left")
    counts["events_per_1000"] = counts["event_count"] / counts["all_events"] * 1000
    save_csv(counts, STATISTICS / "failure_taxonomy.csv")
    scope = pd.DataFrame(
        [
            {
                "requested_failure_mode": "false low-risk classification",
                "status": "not empirically identifiable",
                "reason": "Severity is a predecision model-derived variable; no independent incident-level normative risk label is available.",
            },
            {
                "requested_failure_mode": "incorrect exemption from R5 mandatory escalation",
                "status": "operationally testable",
                "reason": "Counted when mandatory_case and delegated are both true; expected zero if policy code is internally consistent.",
            },
            {
                "requested_failure_mode": "delegated AI error",
                "status": "operationally testable",
                "reason": "Frozen AI mismatch delegated without completed correction.",
            },
        ]
    )
    save_csv(scope, STATISTICS / "failure_mode_scope.csv")
    return counts, scope


def attention_summary(main_summary: pd.DataFrame) -> pd.DataFrame:
    selected = main_summary.loc[
        main_summary["regime"].isin(["R1", "R3", "R5"])
        & main_summary["workload_multiplier"].isin([1.00, 1.20, 1.50])
    ].copy()
    columns = [
        "regime",
        "workload_multiplier",
        "attention_exclusive_ai_error_share_mean",
        "attention_exclusive_ambiguous_share_mean",
        "attention_exclusive_high_consequence_correct_share_mean",
        "attention_exclusive_low_other_correct_share_mean",
        "review_minutes_per_1000_mean",
    ]
    frame = selected[columns]
    save_csv(frame, STATISTICS / "attention_allocation_representative.csv")
    return frame


def main() -> None:
    main_runs = pd.read_csv(SIM_RAW / "main_workload_runs.csv", low_memory=False)
    main_summary = pd.read_csv(SIM_AGG / "main_workload_summary.csv")
    priority_runs = pd.read_csv(ABLATIONS / "priority_control_runs.csv", low_memory=False)
    review_runs = pd.read_csv(SENSITIVITY / "review_time_runs.csv", low_memory=False)
    ai_runs = pd.read_csv(SENSITIVITY / "ai_reliability_runs.csv", low_memory=False)

    detailed_crossovers(
        main_runs,
        [],
        PAIRS,
        METRICS,
        STATISTICS / "crossover_results_detailed.csv",
        2000,
    )
    detailed_crossovers(
        review_runs,
        ["review_time_multiplier"],
        [("R1", "R5")],
        ["uncorrected_ai_error_exposure", "effective_human_review_rate"],
        STATISTICS / "review_time_crossover_results.csv",
        1000,
    )
    detailed_crossovers(
        ai_runs,
        ["ai_reliability_scenario"],
        [("R1", "R5")],
        ["uncorrected_ai_error_exposure", "effective_human_review_rate"],
        STATISTICS / "ai_reliability_crossover_results.csv",
        1000,
    )
    priority_effects(main_runs, priority_runs)
    delta_summaries()
    ai_capacity_pareto()
    transition_details(main_runs, main_summary)
    failure_taxonomy()
    attention_summary(main_summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "main_raw_rows": len(main_runs),
                "postprocessing_outputs": 12,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
