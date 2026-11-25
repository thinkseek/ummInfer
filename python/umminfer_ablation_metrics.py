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


def visual_form_observation(record):
    """Measure artifact-form completion, not perceptual visual quality."""
    if record.get("schema_version") != 2:
        raise ValueError("visual-form metrics require trajectory schema v2")
    metadata = record.get("ablation")
    if not isinstance(metadata, dict):
        raise ValueError("visual-form record lacks ablation metadata")
    form = metadata.get("visual_form")
    if form not in ("image_only", "video_only", "image_then_i2v_animate"):
        raise ValueError("unknown visual_form in ablation metadata")
    slots = record.get("visual_slots")
    animations = metadata.get("animations")
    if not isinstance(slots, list) or not isinstance(animations, list):
        raise ValueError("visual-form record lacks visual_slots or animations")
    expected_media = "video" if form == "video_only" else "image"
    form_matches = bool(slots) and all(
        slot.get("media_type") == expected_media for slot in slots)
    source_hash_by_slot = {
        artifact.get("slot_id"): artifact.get("sha256")
        for artifact in record.get("artifacts", [])
        if artifact.get("media_type") == "image" and artifact.get("slot_id") is not None}
    expected_source_slots = {slot.get("slot_id") for slot in slots}
    animation_source_slots = [animation.get("source_slot_id") for animation in animations]
    completed_animations = [animation for animation in animations
                            if animation.get("terminal_state") == "completed"
                            and animation.get("source_slot_id") in expected_source_slots
                            and isinstance(animation.get("artifact"), dict)
                            and animation["artifact"].get("media_type") == "video"
                            and animation["artifact"].get("source_slot_id")
                            == animation["source_slot_id"]
                            and animation["artifact"].get("source_image_sha256")
                            == source_hash_by_slot.get(animation["source_slot_id"])]
    if form == "image_then_i2v_animate":
        form_matches = form_matches and len(animations) == len(slots) \
            and len(set(animation_source_slots)) == len(animation_source_slots) \
            and set(animation_source_slots) == expected_source_slots \
            and len(completed_animations) == len(slots)
    else:
        form_matches = form_matches and not animations
    complete = record.get("terminal_state") == "completed" and form_matches
    expected_animation_count = len(slots) if form == "image_then_i2v_animate" else 0
    return {
        "coverage": "COVERED",
        "visual_form": form,
        "completion": 1.0 if complete else 0.0,
        "source_visual_slot_count": len(slots),
        "source_image_count": len(source_hash_by_slot),
        "expected_animation_count": expected_animation_count,
        "completed_animation_count": len(completed_animations),
        "image_text_relevance": None,
        "first_frame_consistency": None,
        "user_blind_preference": None,
        "quality_coverage": "NOT_COVERED",
    }


def aggregate_visual_form(records):
    observations = [visual_form_observation(record) for record in records]
    total = len(observations)
    expected = sum(item["expected_animation_count"] for item in observations)
    completed = sum(item["completed_animation_count"] for item in observations)
    return {
        "coverage": "COVERED" if total else "NOT_COVERED",
        "total_case_count": total,
        "visual_form_completion_rate": (
            sum(item["completion"] for item in observations) / total if total else None),
        "expected_animation_count": expected,
        "completed_animation_count": completed,
        "animation_success_rate": completed / expected if expected else None,
        "animation_coverage": "COVERED" if expected else "NOT_APPLICABLE",
        "image_text_relevance": None,
        "first_frame_consistency": None,
        "user_blind_preference": None,
        "quality_coverage": "NOT_COVERED",
    }


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
