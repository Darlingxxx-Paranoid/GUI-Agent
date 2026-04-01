"""Core Oracle contracts and strict parser utilities."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import types
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints


Decision = Literal["success", "fail", "uncertain"]
Outcome = Literal["pass", "fail", "uncertain"]
Severity = Literal["none", "soft", "hard"]
Scope = Literal["anchor", "local", "global"]
Phase = Literal["pre", "post"]

SelectorKind = Literal[
    "widget_id",
    "resource_id",
    "text",
    "class_name",
    "content_desc",
    "bounds",
    "point",
    "tag",
]

SelectorOperator = Literal[
    "equals",
    "contains",
    "in",
    "exists",
    "regex",
    "overlap",
    "near",
]

PredicateOperator = Literal[
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "in",
    "not_in",
    "any_of",
    "all_of",
    "exists",
    "not_exists",
    "regex",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "overlap",
    "near",
]

PolicyKind = Literal[
    "app_boundary",
    "activity_boundary",
    "risk_boundary",
    "loop_guard",
    "visual_guard",
    "input_guard",
    "focus_guard",
]

ActionType = Literal[
    "tap",
    "input",
    "swipe",
    "back",
    "enter",
    "long_press",
    "launch_app",
]

EventFactType = Literal[
    "package_changed",
    "activity_changed",
    "text_appeared",
    "text_disappeared",
    "widget_appeared",
    "widget_disappeared",
    "keyboard_changed",
    "focus_changed",
    "risk_detected",
    "target_resolved",
    "target_missing",
]

StateFactType = Literal[
    "keyboard_state",
    "focus_state",
    "package_relation_state",
    "target_presence_state",
    "target_region_state",
    "forbidden_package_state",
    "switch_state",
    "switch_checked_delta_state",
    "visual_similarity_state",
]

FactType = Union[EventFactType, StateFactType]

ObservationSource = Literal[
    "ui_tree",
    "screenshot_diff",
    "ocr",
    "cv",
    "runtime_guard",
    "action_mapper",
    "policy_engine",
    "semantic_llm",
]

MetricKind = Literal["score", "ratio", "count", "duration_ms", "similarity"]


UNION_ORIGINS = (Union,)
if hasattr(types, "UnionType"):
    UNION_ORIGINS = (Union, types.UnionType)


ACTION_TYPES = {
    "tap",
    "input",
    "swipe",
    "back",
    "enter",
    "long_press",
    "launch_app",
}

POLICY_KINDS = {
    "app_boundary",
    "activity_boundary",
    "risk_boundary",
    "loop_guard",
    "visual_guard",
    "input_guard",
    "focus_guard",
}

OBSERVATION_SOURCES = {
    "ui_tree",
    "screenshot_diff",
    "ocr",
    "cv",
    "runtime_guard",
    "action_mapper",
    "policy_engine",
    "semantic_llm",
}

FACT_ATTRIBUTE_SCHEMA: dict[str, frozenset[str]] = {
    # Event facts
    "package_changed": frozenset({"attributes.old", "attributes.new"}),
    "activity_changed": frozenset({"attributes.old", "attributes.new"}),
    "text_appeared": frozenset({"attributes.text", "attributes.from_action_input"}),
    "text_disappeared": frozenset({"attributes.text"}),
    "widget_appeared": frozenset(
        {
            "attributes.widget_id",
            "attributes.resource_id",
            "attributes.text",
            "attributes.content_desc",
            "attributes.bounds",
            "attributes.center",
        }
    ),
    "widget_disappeared": frozenset(
        {
            "attributes.widget_id",
            "attributes.resource_id",
            "attributes.text",
            "attributes.content_desc",
            "attributes.bounds",
            "attributes.center",
        }
    ),
    "keyboard_changed": frozenset({"attributes.old", "attributes.new"}),
    "focus_changed": frozenset({"attributes.old", "attributes.new"}),
    "risk_detected": frozenset({"attributes.risk", "attributes.history_len", "attributes.message"}),
    "target_resolved": frozenset({"attributes.widget_id", "attributes.bounds", "attributes.center"}),
    "target_missing": frozenset({"attributes.reason"}),
    # State facts
    "keyboard_state": frozenset({"attributes.visible"}),
    "focus_state": frozenset({"attributes.focused_widget", "attributes.widget_id", "attributes.focused"}),
    "package_relation_state": frozenset({"attributes.old", "attributes.new", "attributes.relation"}),
    "target_presence_state": frozenset({"attributes.present"}),
    "target_region_state": frozenset({"attributes.center", "attributes.bounds"}),
    "forbidden_package_state": frozenset({"attributes.package", "attributes.forbidden"}),
    "switch_state": frozenset(
        {
            "attributes.old",
            "attributes.new",
            "attributes.changed",
            "attributes.enabled",
            "attributes.checked",
        }
    ),
    "switch_checked_delta_state": frozenset(
        {"attributes.old", "attributes.new", "attributes.delta", "attributes.changed"}
    ),
    "visual_similarity_state": frozenset({"attributes.similarity"}),
}


@dataclass
class GoalSpec:
    summary: str
    success_definition: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Selector:
    kind: Union[SelectorKind, str]
    operator: Union[SelectorOperator, str]
    value: Any


@dataclass
class TargetResolved:
    widget_id: Optional[int] = None
    bounds: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    snapshot: Optional[dict] = None


@dataclass
class TargetRef:
    ref_id: Optional[str] = None
    role: Optional[Literal["primary", "secondary", "fallback"]] = None
    selectors: list[Selector] = field(default_factory=list)
    resolved: Optional[TargetResolved] = None


@dataclass
class Predicate:
    field: str
    operator: PredicateOperator
    value: Any = None


@dataclass
class Expectation:
    id: str
    fact_type: Union[FactType, str]
    scope: Optional[Scope] = None
    subject_ref: Optional[str] = None
    predicates: list[Predicate] = field(default_factory=list)
    weight: float = 1.0
    optional: bool = False
    polarity: Literal["positive", "negative"] = "positive"
    tier: Literal["required", "supporting"] = "required"


@dataclass
class PolicyRule:
    id: str
    kind: Union[PolicyKind, str]
    level: Severity = "soft"
    subject_ref: Optional[str] = None
    predicates: list[Predicate] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    boundary_mode: Optional[Literal["stay", "switch", "either"]] = None
    expected_packages: list[str] = field(default_factory=list)
    forbidden_packages: list[str] = field(default_factory=list)
    expected_activity_contains: Optional[str] = None
    loop_threshold: Optional[int] = None
    max_similarity: Optional[float] = None


@dataclass
class StepContract:
    goal: GoalSpec
    target: Optional[TargetRef] = None
    expectations: list[Expectation] = field(default_factory=list)
    policies: list[PolicyRule] = field(default_factory=list)
    planning_hints: dict = field(default_factory=dict)


@dataclass
class ResolvedAction:
    type: Union[ActionType, str]
    params: dict = field(default_factory=dict)
    target: Optional[TargetRef] = None
    description: str = ""


@dataclass
class ObservationFact:
    fact_id: str
    type: Union[FactType, str]
    scope: Optional[Scope] = None
    subject_ref: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    confidence: Optional[float] = None
    source: Optional[Union[ObservationSource, str]] = None
    is_derived: bool = False
    evidence_refs: Optional[list[str]] = None


@dataclass
class AdviceParams:
    retry_count: Optional[int] = None
    observe_delay_ms: Optional[int] = None
    backtrack_steps: Optional[int] = None
    reason_tags: Optional[list[str]] = None
    target_ref: Optional[str] = None


@dataclass
class Assessment:
    name: str
    source: Literal["runtime", "policy_engine", "evaluator", "semantic_llm"]
    applies_to: Literal["boundary", "effect", "semantic", "safety", "runtime_guard"]
    outcome: Outcome
    severity: Severity
    reason_code: Optional[str] = None
    message: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    score: Optional[float] = None
    remedy_hint: Optional[AdviceParams] = None


@dataclass
class GuardResult:
    phase: Phase
    allowed: bool
    assessments: list[Assessment] = field(default_factory=list)
    observations: list[ObservationFact] = field(default_factory=list)


@dataclass
class NumericMetric:
    key: str
    kind: MetricKind
    value: float
    unit: Optional[str] = None


@dataclass
class ExpectationMatch:
    expectation_id: str
    matched: bool
    matched_fact_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    reason_code: Optional[str] = None
    message: str = ""


@dataclass
class RecommendedAction:
    kind: Literal["continue", "retry", "replan", "backtrack", "observe", "abort"]
    params: Optional[AdviceParams] = None


@dataclass
class StepEvaluation:
    decision: Decision
    confidence: float
    recommended_action: Optional[RecommendedAction] = None
    assessments: list[Assessment] = field(default_factory=list)
    observations: list[ObservationFact] = field(default_factory=list)
    metrics: list[NumericMetric] = field(default_factory=list)
    expectation_matches: list[ExpectationMatch] = field(default_factory=list)


class ContractValidationError(ValueError):
    """Raised when contract payload does not match dataclass schema."""


T = TypeVar("T")


def normalize_subject_ref(subject_ref: str | None, fallback_widget_id: int | None = None) -> str:
    if subject_ref:
        subject_ref = str(subject_ref).strip()
        if subject_ref:
            return subject_ref
    if fallback_widget_id is not None:
        return f"widget:{int(fallback_widget_id)}"
    return "target:primary"


def to_plain_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        output: dict[str, Any] = {}
        for item in fields(value):
            output[item.name] = to_plain_dict(getattr(value, item.name))
        return output
    if isinstance(value, dict):
        return {str(k): to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_dict(v) for v in value]
    return value


def _type_name(tp: Any) -> str:
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def parse_dataclass(payload: Any, cls: Type[T], strict: bool = True) -> T:
    if not is_dataclass(cls):
        raise ContractValidationError(f"{cls} is not a dataclass type")
    if not isinstance(payload, dict):
        raise ContractValidationError(f"Expected dict for {cls.__name__}, got {type(payload).__name__}")

    allowed = {item.name for item in fields(cls)}
    if strict:
        extras = sorted(set(payload.keys()) - allowed)
        if extras:
            raise ContractValidationError(f"Unexpected fields for {cls.__name__}: {extras}")

    type_hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        field_type = type_hints.get(item.name, item.type)
        if item.name in payload:
            kwargs[item.name] = _parse_value(payload[item.name], field_type, strict)
            continue

        if item.default is not MISSING:
            kwargs[item.name] = copy.deepcopy(item.default)
            continue

        if item.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            kwargs[item.name] = item.default_factory()  # type: ignore[misc]
            continue

        if _is_optional(field_type):
            kwargs[item.name] = None
            continue

        raise ContractValidationError(f"Missing required field for {cls.__name__}: {item.name}")

    return cls(**kwargs)


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in UNION_ORIGINS:
        return any(arg is type(None) for arg in args)
    return False


def _parse_value(value: Any, tp: Any, strict: bool) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return value

    if origin in UNION_ORIGINS:
        last_error: Optional[Exception] = None
        for arg in args:
            if arg is type(None):
                if value is None:
                    return None
                continue
            try:
                return _parse_value(value, arg, strict)
            except Exception as exc:  # pragma: no cover - best effort union parse
                last_error = exc
                continue
        if value is None:
            return None
        raise ContractValidationError(f"Value {value!r} does not match union {_type_name(tp)}: {last_error}")

    if origin is Literal:
        literals = set(args)
        if value not in literals:
            raise ContractValidationError(f"Expected one of {sorted(literals)}, got {value!r}")
        return value

    if is_dataclass(tp):
        return parse_dataclass(value, tp, strict=strict)

    if origin in (list, List):
        if not isinstance(value, list):
            raise ContractValidationError(f"Expected list for {_type_name(tp)}, got {type(value).__name__}")
        inner = args[0] if args else Any
        return [_parse_value(v, inner, strict) for v in value]

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ContractValidationError(f"Expected dict for {_type_name(tp)}, got {type(value).__name__}")
        key_tp = args[0] if len(args) > 0 else Any
        val_tp = args[1] if len(args) > 1 else Any
        out: dict[Any, Any] = {}
        for key, val in value.items():
            parsed_key = _parse_value(key, key_tp, strict) if key_tp is not Any else key
            out[parsed_key] = _parse_value(val, val_tp, strict)
        return out

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise ContractValidationError(f"Expected tuple/list for {_type_name(tp)}, got {type(value).__name__}")
        raw = list(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_parse_value(v, args[0], strict) for v in raw)
        if args and len(raw) != len(args):
            raise ContractValidationError(f"Tuple length mismatch for {_type_name(tp)}: {len(raw)}")
        if not args:
            return tuple(raw)
        return tuple(_parse_value(raw[idx], args[idx], strict) for idx in range(len(args)))

    if tp is str:
        if not isinstance(value, str):
            raise ContractValidationError(f"Expected str, got {type(value).__name__}")
        return value

    if tp is bool:
        if not isinstance(value, bool):
            raise ContractValidationError(f"Expected bool, got {type(value).__name__}")
        return value

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(f"Expected int, got {type(value).__name__}")
        return value

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError(f"Expected float, got {type(value).__name__}")
        return float(value)

    if isinstance(tp, type):
        if not isinstance(value, tp):
            raise ContractValidationError(f"Expected {tp.__name__}, got {type(value).__name__}")
        return value

    return value


def dataclass_to_json_schema(cls: Type[Any]) -> dict:
    if not is_dataclass(cls):
        raise ContractValidationError(f"{cls} is not a dataclass")
    type_hints = get_type_hints(cls)
    return {
        "title": cls.__name__,
        "type": "object",
        "properties": {
            item.name: _annotation_to_schema(type_hints.get(item.name, item.type)) for item in fields(cls)
        },
        "required": [
            item.name
            for item in fields(cls)
            if item.default is MISSING
            and item.default_factory is MISSING
            and not _is_optional(type_hints.get(item.name, item.type))
        ],
        "additionalProperties": False,
    }


def _annotation_to_schema(tp: Any) -> dict:
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return {}

    if origin in UNION_ORIGINS:
        parts = []
        for arg in args:
            if arg is type(None):
                parts.append({"type": "null"})
            else:
                parts.append(_annotation_to_schema(arg))
        return {"anyOf": parts}

    if origin is Literal:
        values = list(args)
        literals = [v for v in values if v is not None]
        schema: dict[str, Any] = {"enum": literals}
        if any(v is None for v in values):
            schema = {"anyOf": [schema, {"type": "null"}]}
        return schema

    if is_dataclass(tp):
        return dataclass_to_json_schema(tp)

    if origin in (list, List):
        inner = args[0] if args else Any
        return {"type": "array", "items": _annotation_to_schema(inner)}

    if origin in (dict, Dict):
        val_tp = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": _annotation_to_schema(val_tp)}

    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _annotation_to_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [_annotation_to_schema(arg) for arg in args],
            "minItems": len(args),
            "maxItems": len(args),
        }

    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}

    return {}


def contract_schema_bundle() -> dict:
    classes = [
        GoalSpec,
        Selector,
        TargetResolved,
        TargetRef,
        Predicate,
        Expectation,
        PolicyRule,
        StepContract,
        ResolvedAction,
        ObservationFact,
        AdviceParams,
        Assessment,
        GuardResult,
        NumericMetric,
        ExpectationMatch,
        RecommendedAction,
        StepEvaluation,
    ]
    return {
        "schema_version": "v4.0",
        "definitions": {cls.__name__: dataclass_to_json_schema(cls) for cls in classes},
    }


def export_contract_schema(path: str) -> str:
    bundle = contract_schema_bundle()
    with open(path, "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2)
    return path


def parse_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

    # tolerate leading/trailing non-json text.
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"JSON parse failed: {exc}") from exc

    if not isinstance(data, dict):
        raise ContractValidationError("Top-level JSON must be an object")
    return data


def validate_action_type(action_type: str, allow_experimental: bool = False) -> str:
    value = str(action_type or "").strip().lower()
    if value in ACTION_TYPES:
        return value
    if allow_experimental and value.startswith("x_"):
        return value
    raise ContractValidationError(f"Invalid action type: {action_type!r}")


def sanitize_reason_tags(tags: Optional[list[str]]) -> Optional[list[str]]:
    if tags is None:
        return None
    out: list[str] = []
    for tag in tags:
        value = re.sub(r"[^a-zA-Z0-9_\-:.]", "", str(tag or "").strip())
        if value:
            out.append(value)
    return out or None
