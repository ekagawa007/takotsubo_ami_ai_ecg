"""Lightweight inference progress on stderr, independent of ML libraries."""

import json
import shutil
import sys
import time


JSON_PREFIX = "ECG_PROGRESS "
_DETAIL_FIELDS = ("model", "model_index", "model_count", "tta_index", "tta_total",
                  "images_completed", "images_total")
_MODEL_LABELS = {"vit": "ViT", "convnextv2": "ConvNeXtV2", "mlpmixer": "MLP-Mixer"}


class ProgressReporter:
    """Consume progress dictionaries without changing inference state or RNG.

    Counts refer to completed image-views (images times TTA views). Indices are
    one-based. JSON events are complete snapshots; a stage clears stale model,
    TTA and image details unless those details are explicitly supplied.
    """

    def __init__(self, mode="auto", stream=None, clock=None):
        if mode not in ("auto", "bar", "json", "off"):
            raise ValueError("Unknown progress mode.")
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        self.started = self.clock()
        self.mode = ("bar" if self.stream.isatty() else "plain") if mode == "auto" else mode
        self.state = {"completed": 0, "total": 0, **dict.fromkeys(_DETAIL_FIELDS)}
        self.last_progress_time = None
        self.bar_open = False
        self.bar_width = 0

    def __call__(self, event):
        kind = event.get("event", "progress")
        if kind not in ("stage", "progress", "complete", "error"):
            raise ValueError("Unknown progress event.")
        now = self.clock()
        state = dict(self.state)
        if kind != "progress":
            state.update(dict.fromkeys(_DETAIL_FIELDS))
        for key in ("completed", "total", *_DETAIL_FIELDS):
            if key in event:
                state[key] = event[key]
        self.state = state
        record = {"event": kind, "message": event.get("message", "Running inference."),
                  **state, "elapsed_seconds": max(0.0, now - self.started)}
        if self.mode == "off":
            return
        if kind == "progress":
            last = ((state["total"] > 0 and state["completed"] >= state["total"])
                    or (state["tta_total"] is not None and state["tta_index"] == state["tta_total"]
                        and state["images_total"] is not None and state["images_completed"] == state["images_total"]))
            interval = 1.0 if self.mode == "plain" else 0.2
            if self.last_progress_time is not None and not last and now - self.last_progress_time < interval:
                return
            self.last_progress_time = now
        else:
            self.last_progress_time = None
        if self.mode == "json":
            self.stream.write(JSON_PREFIX + json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        else:
            line = self._format(record)
            if self.mode == "bar":
                previous_width = min(self.bar_width, self._bar_limit())
                self.stream.write("\r" + line + " " * max(0, previous_width - len(line)))
                self.bar_width = len(line)
                self.bar_open = True
                if kind in ("complete", "error"):
                    self.stream.write("\n")
                    self.bar_open = False
            else:
                self.stream.write(line + "\n")
        self.stream.flush()

    def _format(self, event):
        if self.mode == "bar":
            return self._format_bar(event)
        completed, total = event["completed"], event["total"]
        parts = [event["message"]]
        if total:
            fraction = max(0.0, min(1.0, completed / total))
            parts.append(f"{100 * fraction:.1f}% ({completed}/{total} image-views)")
        if event["model"] is not None:
            label = _MODEL_LABELS.get(event["model"], event["model"])
            position = (f" {event['model_index']}/{event['model_count']}"
                        if event["model_index"] is not None and event["model_count"] is not None else "")
            parts.append(f"model{position}: {label}")
        if event["tta_index"] is not None and event["tta_total"] is not None:
            parts.append(f"TTA view {event['tta_index']}/{event['tta_total']}")
        if event["images_completed"] is not None and event["images_total"] is not None:
            parts.append(f"images {event['images_completed']}/{event['images_total']}")
        parts.append(f"elapsed {event['elapsed_seconds']:.1f}s")
        return " | ".join(parts)

    @staticmethod
    def _bar_limit():
        # Leave one column free to avoid the terminal's automatic line wrap.
        return max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)

    def _format_bar(self, event):
        parts = []
        if event["total"]:
            fraction = max(0.0, min(1.0, event["completed"] / event["total"]))
            filled = int(fraction * 20)
            parts.append("[" + "#" * filled + "-" * (20 - filled) + f"] {100 * fraction:.1f}%")
        messages = {"Preparing inputs and output.": "Preparing", "Initializing inference libraries.": "Initializing",
                    "Checking inputs and checkpoints.": "Checking inputs", "Loading model.": "Loading",
                    "Running inference.": "Running", "Combining predictions.": "Combining",
                    "Saving predictions.": "Saving", "Predictions saved.": "Saved",
                    "Inference completed.": "Completed", "Inference failed.": "Failed",
                    "Inference interrupted.": "Interrupted"}
        if event["message"] != "Running inference." or event["model"] is None:
            parts.append(messages.get(event["message"], event["message"].rstrip(".")))
        if event["model"] is not None:
            parts.append(_MODEL_LABELS.get(event["model"], event["model"]))
            if event["model_index"] is not None and event["model_count"] is not None:
                parts.append(f"{event['model_index']}/{event['model_count']}")
        if event["tta_index"] is not None and event["tta_total"] is not None:
            parts.append(f"TTA {event['tta_index']}/{event['tta_total']}")
        if event["images_completed"] is not None and event["images_total"] is not None:
            parts.append(f"{event['images_completed']}/{event['images_total']}")
        elapsed = int(event["elapsed_seconds"])
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        parts.append(f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}")
        line = " ".join(parts)
        limit = self._bar_limit()
        return line if len(line) <= limit else (line[:max(0, limit - 3)] + "..." if limit >= 3 else line[:limit])

    def stage(self, message, **fields):
        self({"event": "stage", "message": message, **fields})

    def complete(self, message="Inference completed."):
        self({"event": "complete", "message": message})

    def error(self, message="Inference failed."):
        self({"event": "error", "message": message})

    def close(self):
        if self.bar_open:
            self.stream.write("\n")
            self.stream.flush()
            self.bar_open = False
