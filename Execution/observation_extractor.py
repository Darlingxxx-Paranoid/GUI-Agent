"""Observation extraction for runtime and post-evaluation."""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from Oracle.contracts import ObservationFact, ResolvedAction, TargetRef, normalize_subject_ref
from Perception.context_builder import UIState, WidgetInfo


class ObservationExtractor:
    def __init__(self):
        self._counter = 0

    def extract(
        self,
        old_state: UIState,
        new_state: UIState,
        action: ResolvedAction,
    ) -> list[ObservationFact]:
        facts: list[ObservationFact] = []

        old_pkg = str(getattr(old_state, "package_name", "") or "")
        new_pkg = str(getattr(new_state, "package_name", "") or "")
        old_act = str(getattr(old_state, "activity_name", "") or "")
        new_act = str(getattr(new_state, "activity_name", "") or "")

        if old_pkg and new_pkg and old_pkg != new_pkg:
            facts.append(
                self._fact(
                    fact_type="package_changed",
                    scope="global",
                    subject_ref=None,
                    attributes={"old": old_pkg, "new": new_pkg},
                    source="ui_tree",
                )
            )

        facts.append(
            self._fact(
                fact_type="package_relation_state",
                scope="global",
                subject_ref=None,
                attributes={
                    "old": old_pkg,
                    "new": new_pkg,
                    "relation": "same" if old_pkg == new_pkg else "changed",
                },
                source="ui_tree",
            )
        )

        if old_act and new_act and old_act != new_act:
            facts.append(
                self._fact(
                    fact_type="activity_changed",
                    scope="global",
                    subject_ref=None,
                    attributes={"old": old_act, "new": new_act},
                    source="ui_tree",
                )
            )

        old_keyboard = bool(getattr(old_state, "keyboard_visible", False))
        new_keyboard = bool(getattr(new_state, "keyboard_visible", False))
        if old_keyboard != new_keyboard:
            facts.append(
                self._fact(
                    fact_type="keyboard_changed",
                    scope="global",
                    subject_ref=None,
                    attributes={"old": old_keyboard, "new": new_keyboard},
                    source="ui_tree",
                )
            )
        facts.append(
            self._fact(
                fact_type="keyboard_state",
                scope="global",
                subject_ref=None,
                attributes={"visible": new_keyboard},
                source="ui_tree",
            )
        )

        old_focus = self._focused_widget_signature(old_state)
        new_focus = self._focused_widget_signature(new_state)
        if old_focus != new_focus:
            facts.append(
                self._fact(
                    fact_type="focus_changed",
                    scope="global",
                    subject_ref=None,
                    attributes={"old": old_focus, "new": new_focus},
                    source="ui_tree",
                )
            )
        facts.append(
            self._fact(
                fact_type="focus_state",
                scope="global",
                subject_ref=None,
                attributes={"focused_widget": new_focus},
                source="ui_tree",
            )
        )

        old_texts = self._collect_text_tokens(old_state)
        new_texts = self._collect_text_tokens(new_state)
        appeared = sorted(new_texts - old_texts)
        disappeared = sorted(old_texts - new_texts)

        for token in appeared[:3]:
            facts.append(
                self._fact(
                    fact_type="text_appeared",
                    scope="global",
                    subject_ref=None,
                    attributes={"text": token},
                    source="ui_tree",
                )
            )

        for token in disappeared[:3]:
            facts.append(
                self._fact(
                    fact_type="text_disappeared",
                    scope="global",
                    subject_ref=None,
                    attributes={"text": token},
                    source="ui_tree",
                )
            )

        text_input = str((action.params or {}).get("text") or "").strip()
        if text_input and any(text_input.lower() in token.lower() for token in new_texts):
            facts.append(
                self._fact(
                    fact_type="text_appeared",
                    scope="local",
                    subject_ref=normalize_subject_ref(action.target.ref_id if action.target else None),
                    attributes={"text": text_input[:120], "from_action_input": True},
                    source="ui_tree",
                )
            )

        similarity = self._textual_similarity(old_texts, new_texts)
        facts.append(
            self._fact(
                fact_type="visual_similarity_state",
                scope="local",
                subject_ref=normalize_subject_ref(action.target.ref_id if action.target else None),
                attributes={"similarity": similarity},
                source="screenshot_diff",
            )
        )

        self._append_target_facts(facts=facts, action=action, new_state=new_state)
        return facts

    def _append_target_facts(self, facts: list[ObservationFact], action: ResolvedAction, new_state: UIState) -> None:
        target = action.target
        subject_ref = normalize_subject_ref(target.ref_id if target else None)
        if target is None:
            return

        target_found = self._find_target_in_state(target=target, state=new_state)
        if target_found is not None:
            facts.append(
                self._fact(
                    fact_type="target_resolved",
                    scope="anchor",
                    subject_ref=subject_ref,
                    attributes={"widget_id": int(target_found.widget_id)},
                    source="action_mapper",
                )
            )
        else:
            facts.append(
                self._fact(
                    fact_type="target_missing",
                    scope="anchor",
                    subject_ref=subject_ref,
                    attributes={"reason": "target_not_found_after_action"},
                    source="action_mapper",
                )
            )

        facts.append(
            self._fact(
                fact_type="target_presence_state",
                scope="anchor",
                subject_ref=subject_ref,
                attributes={"present": target_found is not None},
                source="ui_tree",
            )
        )

        facts.append(
            self._fact(
                fact_type="target_region_state",
                scope="anchor",
                subject_ref=subject_ref,
                attributes={
                    "center": list(target_found.center) if target_found is not None else None,
                    "bounds": list(target_found.bounds) if target_found is not None else None,
                },
                source="ui_tree",
            )
        )

    def _find_target_in_state(self, target: TargetRef, state: UIState) -> Optional[WidgetInfo]:
        widgets = state.get_runtime_widgets()
        if target.resolved and target.resolved.widget_id is not None:
            by_id = state.find_widget_by_id(int(target.resolved.widget_id))
            if by_id is not None:
                return by_id

        for selector in target.selectors:
            kind = str(selector.kind or "").lower()
            value = selector.value
            if kind == "widget_id":
                try:
                    widget = state.find_widget_by_id(int(value))
                except Exception:
                    widget = None
                if widget is not None:
                    return widget
            if kind == "text":
                widget = state.find_widget_by_text(str(value or ""))
                if widget is not None:
                    return widget
            if kind == "resource_id":
                needle = str(value or "").strip().lower()
                for widget in widgets:
                    if needle and needle in str(widget.resource_id or "").lower():
                        return widget
            if kind == "content_desc":
                needle = str(value or "").strip().lower()
                for widget in widgets:
                    if needle and needle in str(widget.content_desc or "").lower():
                        return widget
        return None

    def _focused_widget_signature(self, state: UIState) -> str:
        for widget in state.get_runtime_widgets():
            if getattr(widget, "focused", False):
                token = "|".join(
                    [
                        str(widget.widget_id),
                        str(widget.resource_id or ""),
                        str(widget.text or ""),
                        str(widget.content_desc or ""),
                    ]
                )
                return token
        return ""

    def _collect_text_tokens(self, state: UIState) -> set[str]:
        out: set[str] = set()
        for widget in state.get_runtime_widgets():
            for token in (widget.text, widget.content_desc):
                value = str(token or "").strip()
                if value:
                    out.add(value)
        return out

    def _textual_similarity(self, old_values: set[str], new_values: set[str]) -> float:
        if not old_values and not new_values:
            return 1.0
        union = old_values.union(new_values)
        if not union:
            return 1.0
        inter = old_values.intersection(new_values)
        return round(len(inter) / float(len(union)), 4)

    def _fact(
        self,
        fact_type: str,
        scope: str,
        subject_ref: str | None,
        attributes: dict,
        source: str,
    ) -> ObservationFact:
        self._counter += 1
        digest = hashlib.md5(
            (
                f"{self._counter}|{fact_type}|{scope}|{subject_ref}|{attributes}|{source}"
            ).encode("utf-8")
        ).hexdigest()[:8]
        return ObservationFact(
            fact_id=f"obs_{self._counter:04d}_{digest}",
            type=fact_type,
            scope=scope,
            subject_ref=subject_ref,
            attributes=dict(attributes or {}),
            confidence=1.0,
            source=source,
            is_derived=False,
            evidence_refs=None,
        )
