# SPDX-License-Identifier: Apache-2.0
"""Pluggable, export-only rewards for recorded UMM trajectories.

A reward reads only a recorded trajectory export (the JSONL records written
by umminfer_rl_trajectory), never serving-process memory. A missing score is a
first-class outcome: NullReward returns None and the record keeps
``score: null`` so downstream consumers distinguish "reward missing" from
"reward zero". CompositeUmmReward adds a deterministic multi-objective
experiment signal without changing that coverage rule.
"""

import importlib
import json
import math
from pathlib import Path

from umminfer_sse_replay import validate_replay_metadata

REWARD_SCHEMA_VERSION = 1


class Reward:
    """Maps one trajectory record to a float score, or None if not scorable."""

    #: Stable name written into the record so scores stay interpretable.
    name = "abstract"

    def score(self, record):
        raise NotImplementedError

    def evaluate(self, record):
        """Return ``(score, optional_components)`` for export annotation."""
        return self.score(record), None


class NullReward(Reward):
    """Default reward: scoring is not configured. Never invents a number."""

    name = "null"

    def score(self, record):
        return None


class CompositeUmmReward(Reward):
    """Deterministic multi-objective UMM reward from one schema-v2 record.

    Required analysis metadata stays inside the exported JSON object::

        record["ablation"]["expected_visual_modalities"] = ["image"]
        record["ablation"]["latency"] = {"ttft_ms": 200, "terminal_ms": 1800}

    The default score is ``0.4 * routing_f1 + 0.3 * prose_visual_balance +
    0.3 * barge_in_stability - 0.2 * latency_penalty``. Each component is in
    [0, 1]. Missing labels or latency observations yield None, never a
    fabricated zero-latency or perfect-routing score.
    """

    name = "umm-composite-multiobjective-v1"
    DEFAULT_WEIGHTS = {
        "routing_f1": 0.4,
        "prose_visual_balance": 0.3,
        "barge_in_stability": 0.3,
        "latency_penalty": 0.2,
    }
    DEFAULT_LATENCY_BUDGET_MS = {
        "ttft_ms": 1000.0,
        "ttfa_ms": 1500.0,
        "terminal_ms": 30000.0,
        "barge_in_ms": 250.0,
    }
    _VISUAL_MODALITIES = frozenset(("image", "video"))
    _BARGE_IN_EVENTS = frozenset(
        ("input_audio_buffer.speech_started", "speech_started"))

    def __init__(self, *, weights=None, latency_budget_ms=None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            if not isinstance(weights, dict) \
                    or not set(weights).issubset(self.weights):
                raise ValueError("weights contains an unknown objective")
            self.weights.update(weights)
        for name, value in self.weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"weight {name} must be finite and non-negative")
        if sum(self.weights[name] for name in
               ("routing_f1", "prose_visual_balance", "barge_in_stability")) <= 0:
            raise ValueError("at least one positive objective weight is required")

        self.latency_budget_ms = dict(self.DEFAULT_LATENCY_BUDGET_MS)
        if latency_budget_ms is not None:
            if not isinstance(latency_budget_ms, dict) \
                    or not set(latency_budget_ms).issubset(self.latency_budget_ms):
                raise ValueError("latency_budget_ms contains an unknown observation")
            self.latency_budget_ms.update(latency_budget_ms)
        for name, value in self.latency_budget_ms.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"latency budget {name} must be finite and positive")

    def _routing(self, record):
        slots = record.get("visual_slots")
        if not isinstance(slots, list):
            raise ValueError("schema-v2 record lacks visual_slots")
        predicted = []
        for slot in slots:
            if not isinstance(slot, dict) \
                    or slot.get("media_type") not in self._VISUAL_MODALITIES:
                raise ValueError("visual slot has an unsupported modality")
            predicted.append(slot["media_type"])
        metadata = record.get("ablation")
        if not isinstance(metadata, dict) \
                or "expected_visual_modalities" not in metadata:
            return predicted, None, None
        expected = metadata["expected_visual_modalities"]
        if not isinstance(expected, list) or any(
                modality not in self._VISUAL_MODALITIES for modality in expected):
            raise ValueError(
                "expected_visual_modalities must contain only image/video labels")
        prediction_set = set(predicted)
        expected_set = set(expected)
        true_positive = len(prediction_set & expected_set)
        denominator = (2 * true_positive + len(prediction_set - expected_set)
                       + len(expected_set - prediction_set))
        f1 = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
        return predicted, list(expected), f1

    def _latency(self, record):
        metadata = record["ablation"]
        observed = metadata.get("latency")
        if observed is None:
            return None, None
        if not isinstance(observed, dict):
            raise ValueError("ablation.latency must be an object")
        if not set(observed).issubset(self.latency_budget_ms):
            raise ValueError("ablation.latency contains an unknown observation")
        values = {}
        penalties = []
        for name, budget in self.latency_budget_ms.items():
            value = observed.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"latency {name} must be finite and non-negative")
            values[name] = float(value)
            penalties.append(min(1.0, float(value) / float(budget)))
        if not penalties:
            return None, None
        return values, sum(penalties) / len(penalties)

    def _barge_in_stability(self, record):
        events = record.get("events")
        if not isinstance(events, list):
            raise ValueError("schema-v2 record lacks events")
        observed = any(event.get("type") in self._BARGE_IN_EVENTS
                       for event in events if isinstance(event, dict))
        state = record.get("terminal_state")
        if not observed:
            # No interruption is neutral only for a clean completed turn. A
            # failed or unexplained cancelled turn does not earn stability.
            return False, 1.0 if state == "completed" else 0.0
        terminal_legs = record.get("terminal_legs")
        if not isinstance(terminal_legs, list):
            raise ValueError("schema-v2 record lacks terminal_legs")
        failed_leg = any(isinstance(leg, dict) and leg.get("state") == "failed"
                         for leg in terminal_legs)
        error_event = any(isinstance(event, dict) and event.get("type") == "error"
                          for event in events)
        stable = state == "cancelled" and not failed_leg and not error_event
        return True, 1.0 if stable else 0.0

    @staticmethod
    def _balance(record, predicted, expected):
        clean_text = record.get("clean_text")
        if not isinstance(clean_text, str):
            raise ValueError("schema-v2 record lacks clean_text")
        prose = 1.0 if clean_text.strip() else 0.0
        predicted_count = len(predicted)
        expected_count = len(expected)
        largest = max(predicted_count, expected_count)
        visual_count = 1.0 if largest == 0 else (
            min(predicted_count, expected_count) / largest)
        if prose + visual_count == 0:
            return 0.0
        return 2.0 * prose * visual_count / (prose + visual_count)

    def evaluate(self, record):
        validate_replay_metadata(record)
        if record.get("schema_version") != 2:
            raise ValueError("composite UMM reward requires trajectory schema v2")
        predicted, expected, routing_f1 = self._routing(record)
        if expected is None:
            return None, None
        latency_ms, latency_penalty = self._latency(record)
        if latency_penalty is None:
            return None, None
        balance = self._balance(record, predicted, expected)
        barge_in_observed, barge_in_stability = self._barge_in_stability(record)
        components = {
            "routing_f1": routing_f1,
            "prose_visual_balance": balance,
            "latency_penalty": latency_penalty,
            "barge_in_stability": barge_in_stability,
            "barge_in_observed": barge_in_observed,
            "expected_visual_count": len(expected),
            "predicted_visual_count": len(predicted),
            "latency_ms": latency_ms,
            "latency_budget_ms": dict(self.latency_budget_ms),
            "weights": dict(self.weights),
        }
        score = (self.weights["routing_f1"] * routing_f1
                 + self.weights["prose_visual_balance"] * balance
                 + self.weights["barge_in_stability"] * barge_in_stability
                 - self.weights["latency_penalty"] * latency_penalty)
        return float(score), components

    def score(self, record):
        return self.evaluate(record)[0]


class VerifiableArithmeticReward(Reward):
    """Exact integer-only answers for the bounded online optimizer fixture.

    The prompt id selects a frozen arithmetic identity.  Only exported clean
    text is read; serving state and model internals are never consulted.
    """

    name = "verifiable-integer-arithmetic-v1"
    ANSWERS = {
        "online-arithmetic-739x461": 340679,
        "online-arithmetic-457x683": 312131,
        "online-arithmetic-7389x4621": 34144569,
        "online-arithmetic-4597x6839": 31438883,
        "online-arithmetic-3274x8957": 29325218,
        "online-arithmetic-8642x7351": 63527342,
        "online-arithmetic-6287x5399": 33943513,
        "online-arithmetic-9173x6842": 62761666,
    }

    def score(self, record):
        if record.get("schema_version") != 2:
            raise ValueError("arithmetic reward requires trajectory schema v2")
        metadata = record.get("ablation")
        if not isinstance(metadata, dict) or metadata.get("prompt_id") not in self.ANSWERS:
            return None
        clean_text = record.get("clean_text")
        if not isinstance(clean_text, str):
            raise ValueError("arithmetic reward requires exported clean_text")
        return float(clean_text.strip() == str(self.ANSWERS[metadata["prompt_id"]]))


def apply_reward(trajectory_path, reward, *, output_path=None):
    """Score every record in a trajectory JSONL file.

    Writes records back (or to ``output_path``) with a ``reward`` block. A
    None score is stored as null, not 0.0.
    """
    source = Path(trajectory_path)
    target = Path(output_path) if output_path else source
    scored = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        validate_replay_metadata(
            record, artifact_dir=source.parent / "artifacts",
            require_raw=record.get("sse_replay") is not None)
        score, components = reward.evaluate(record)
        record["reward"] = {
            "schema_version": REWARD_SCHEMA_VERSION,
            "name": reward.name,
            "score": score,
        }
        if components is not None:
            record["reward"]["components"] = components
        scored.append(json.dumps(record, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False))
    target.write_text("\n".join(scored) + "\n", encoding="utf-8")
    return target


def load_reward(dotted_path):
    """Load a Reward subclass or zero-arg factory from ``module:attr``.

    A custom business reward ships in its own module and stays out of the
    serving codebase. A bare class is instantiated with no args.
    """
    module_name, attr = dotted_path.split(":", 1)
    obj = getattr(importlib.import_module(module_name), attr)
    if callable(obj):
        return obj()
    return obj
