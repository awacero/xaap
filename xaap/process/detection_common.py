from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class DetectionWindow:
    """
    Unified representation of a detection/trigger time window.

    This object is intentionally agnostic to the source detector, so it can be
    used for STA/LTA and deep-learning outputs.
    """

    trace_id: str
    start_time: Any
    end_time: Any
    station: str
    source: str
    phase: Optional[str] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class PhasePick:
    """
    Unified representation of a phase pick (e.g. P/S).
    """

    trace_id: str
    pick_time: Any
    station: str
    phase: str
    source: str
    probability: Optional[float] = None


def station_from_trace_id(trace_id: str) -> str:
    """
    Returns station code from an ObsPy-style trace id.

    Expected format: NET.STA.LOC.CHA
    If the format is not respected, returns an empty string.
    """

    parts = trace_id.split(".")
    if len(parts) > 1:
        return parts[1]
    return ""


def _time_to_str(value: Any) -> str:
    """
    Converts timestamp-like objects (including UTCDateTime) to a string.
    """

    return str(value)


def window_to_dict(window: DetectionWindow) -> dict:
    """
    Serializes DetectionWindow to a CSV/JSON-friendly dict.
    """

    data = asdict(window)
    data["start_time"] = _time_to_str(window.start_time)
    data["end_time"] = _time_to_str(window.end_time)
    return data


def phase_pick_to_dict(pick: PhasePick) -> dict:
    """
    Serializes PhasePick to a CSV/JSON-friendly dict.
    """

    data = asdict(pick)
    data["pick_time"] = _time_to_str(pick.pick_time)
    return data


def windows_to_dataframe(windows: list[DetectionWindow]) -> pd.DataFrame:
    """
    Converts a list of DetectionWindow objects to a pandas DataFrame.
    """

    return pd.DataFrame([window_to_dict(w) for w in windows])


def phase_picks_to_dataframe(picks: list[PhasePick]) -> pd.DataFrame:
    """
    Converts a list of PhasePick objects to a pandas DataFrame.
    """

    return pd.DataFrame([phase_pick_to_dict(p) for p in picks])
