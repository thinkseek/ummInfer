# SPDX-License-Identifier: Apache-2.0
"""Export-only metrics for the Stage C narrative-routing ablation.

Predictions come from schema-v2 ``visual_slots``. Labels are optional analysis
metadata under ``record["ablation"]["expected_visual_modalities"]``; they are
never sent to serving. Unlabelled records deliberately produce no F1,
redundancy, or reward value.
"""

from collections import Counter

from umminfer_rl_reward import Reward

VISUAL_MODALITIES = ("image", "video")
REWARD_NAME = "narrative-routing-f1-minus-redundancy-v1"


def expected_modalities(record):
    """Return a validated label multiset, or None when the case is unlabelled."""
    metadata = record.get("ablation")
    if not isinstance(metadata, dict) \
            or "expected_visual_modalities" not in metadata:
        return None
    labels = metadata["expected_visual_modalities"]
    if not isinstance(labels, list) or any(
            label not in VISUAL_MODALITIES for label in labels):
        raise ValueError(
            "expected_visual_modalities must contain only image/video labels")
    return list(labels)


def routing_observation(record):
    """Compute one record's set-F1 counts and multiset redundancy from export."""
    if record.get("schema_version") != 2:
        raise ValueError("narrative-routing metrics require trajectory schema v2")
    slots = record.get("visual_slots")
    if not isinstance(slots, list):
        raise ValueError("schema-v2 record lacks visual_slots")
    predicted = []
    for slot in slots:
        modality = slot.get("media_type")
        if modality not in VISUAL_MODALITIES:
            raise ValueError("visual slot has an unsupported modality")
        predicted.append(modality)
    expected = expected_modalities(record)
    prediction_set = set(predicted)
    expected_set = set(expected or [])
    true_positive = len(prediction_set & expected_set)
    false_positive = len(prediction_set - expected_set)
    false_negative = len(expected_set - prediction_set)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 1.0 if denominator == 0 else 2 * true_positive / denominator

    redundant = None
    redundancy_rate = None
    if expected is not None:
        predicted_counts = Counter(predicted)
        expected_counts = Counter(expected)
        matched_slots = sum(min(predicted_counts[modality],
                                expected_counts[modality])
                            for modality in VISUAL_MODALITIES)
        redundant = len(predicted) - matched_slots
        redundancy_rate = redundant / len(predicted) if predicted else 0.0
    return {
        "coverage": "COVERED" if expected is not None else "NOT_COVERED",
        "reason": None if expected is not None else "labelled example required",
        "expected_visual_modalities": expected,
        "predicted_visual_modalities": predicted,
        "true_positive": true_positive if expected is not None else None,
        "false_positive": false_positive if expected is not None else None,
        "false_negative": false_negative if expected is not None else None,
        "f1": f1 if expected is not None else None,
        "predicted_slot_count": len(predicted),
        "redundant_slot_count": redundant,
        "redundant_slot_rate": redundancy_rate,
    }


def aggregate_routing(records):
    """Micro-average set F1 and slot redundancy over labelled records only."""
    observations = [routing_observation(record) for record in records]
    labelled = [item for item in observations if item["coverage"] == "COVERED"]
    total = len(observations)
    coverage = "NOT_COVERED"
    reason = "no labelled examples"
    if labelled:
        coverage = "COVERED" if len(labelled) == total else "PARTIAL"
        reason = None if coverage == "COVERED" else "only labelled cases contribute"
    result = {
        "coverage": coverage,
        "reason": reason,
        "labelled_case_count": len(labelled),
        "total_case_count": total,
        "f1": None,
        "redundant_slot_rate": None,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "predicted_slot_count": 0,
        "redundant_slot_count": 0,
    }
    if not labelled:
        return result
    for key in ("true_positive", "false_positive", "false_negative",
                "predicted_slot_count", "redundant_slot_count"):
        result[key] = sum(item[key] for item in labelled)
    denominator = (2 * result["true_positive"] + result["false_positive"]
                   + result["false_negative"])
    result["f1"] = 1.0 if denominator == 0 else (
        2 * result["true_positive"] / denominator)
    result["redundant_slot_rate"] = (
        result["redundant_slot_count"] / result["predicted_slot_count"]
        if result["predicted_slot_count"] else 0.0)
    return result


class NarrativeRoutingReward(Reward):
    """Diagnostic F1-minus-redundancy reward; unlabelled records stay null.

    This is an export-only experiment signal, not a validated business reward or
    evidence of an optimizer/model improvement.
    """

    name = REWARD_NAME

    def score(self, record):
        observation = routing_observation(record)
        if observation["coverage"] != "COVERED":
            return None
        return float(observation["f1"] - observation["redundant_slot_rate"])
