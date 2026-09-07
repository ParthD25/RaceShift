from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFSource:
    repo_id: str
    purpose: str
    approximate_size: str
    license_note: str


SOURCES = [
    HFSource(
        repo_id="FlorindoDev/f1_corner_telemetry_2024_2025",
        purpose="Corner-level 50-sample telemetry sequences for self-supervised/corner modeling",
        approximate_size="~7.1 GB CSV plus normalized archive",
        license_note="Dataset card lists CC BY 4.0; verify upstream telemetry terms before redistribution.",
    ),
    HFSource(
        repo_id="VforVitorio/f1-strategy-dataset",
        purpose="Prepared 2023-2025 lap/telemetry/strategy features for model prototyping",
        approximate_size="~16.1 GB",
        license_note="Dataset card lists Apache-2.0; raw upstream F1 data retains upstream terms.",
    ),
    HFSource(
        repo_id="tobil/imsa",
        purpose="Endurance-racing extension with laps, sectors, stints, classes, pits and weather",
        approximate_size="~264 MB",
        license_note="Dataset card listed MIT at time of project research; re-check before use.",
    ),
]
