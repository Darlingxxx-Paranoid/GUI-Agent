import argparse
import json
import os

from Perception.context_builder import UIState, WidgetInfo


def build_uistate_from_context_json(context_json_path: str) -> UIState:
    with open(context_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    widgets = []
    for w in data.get("widgets", []):
        widgets.append(
            WidgetInfo(
                widget_id=int(w.get("id", 0)),
                bounds=tuple(w.get("bounds", [0, 0, 0, 0])),
                center=tuple(w.get("center", [0, 0])),
                category=w.get("category", "") or "",
                text=w.get("text", "") or "",
                resource_id=w.get("resource_id", "") or "",
                class_name=w.get("class", "") or "",
                content_desc=w.get("content_desc", "") or "",
                clickable=bool(w.get("clickable", False)),
                scrollable=bool(w.get("scrollable", False)),
                cv_confidence=float(w.get("cv_confidence", 0.0) or 0.0),
            )
        )

    return UIState(
        widgets=widgets,
        activity_name=data.get("activity_name", "") or "",
        package_name=data.get("package_name", "") or "",
        screen_width=int(data.get("screen_width", 0) or 0),
        screen_height=int(data.get("screen_height", 0) or 0),
        screenshot_path=data.get("screenshot_path", "") or "",
        raw_cv_elements=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshot_path")
    parser.add_argument("context_json_path")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    screenshot_ok = os.path.exists(args.screenshot_path)
    context_ok = os.path.exists(args.context_json_path)
    if not context_ok:
        raise FileNotFoundError(args.context_json_path)

    ui_state = build_uistate_from_context_json(args.context_json_path)
    prompt_text = ui_state.to_prompt_text()

    if args.out:
        out_path = args.out
    else:
        base = os.path.basename(args.context_json_path)
        stem, _ = os.path.splitext(base)
        out_path = os.path.join(os.path.dirname(args.context_json_path), f"{stem}.prompt.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"screenshot_exists={screenshot_ok}\n")
        f.write(f"context_exists={context_ok}\n")
        f.write(f"screenshot_path={args.screenshot_path}\n")
        f.write(f"context_json_path={args.context_json_path}\n")
        f.write("\n")
        f.write(prompt_text)
        f.write("\n")

    print(prompt_text)
    print(f"\nSaved prompt text to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

