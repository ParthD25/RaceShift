from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from raceshift.data.schema import REQUIRED_FORECAST_COLUMNS
from raceshift.features.full_context import RAW_TARGET_COLUMN, build_full_context_table
from raceshift.models.forward_forward_regressor import ForwardForwardRegressor

ARTIFACT_FILES = (
    "model_weights.npz",
    "model_config.json",
    "training_history.json",
    "preprocessor.joblib",
    "feature_contract.json",
    "metrics.json",
)


def missing_artifact_files(directory: str | Path) -> list[str]:
    """Names of required artifact members that are absent. Empty means complete."""
    path = Path(directory)
    return [name for name in ARTIFACT_FILES if not (path / name).is_file()]


def is_blank(value: object) -> bool:
    """True for a missing identity value: null, or a string that is empty once stripped
    (a whitespace-only event or session name is not a usable selector key)."""
    return bool(pd.isna(value)) or str(value).strip() in {"", "nan", "None"}


def blank_mask(values: pd.Series) -> pd.Series:
    """Row mask of missing identity values, see :func:`is_blank`."""
    return values.isna() | values.astype(str).str.strip().isin({"", "nan", "None"})


def integer_season(value: object) -> int | None:
    """The season as an int when the value is a whole number, else None (a fractional or
    non-numeric season is not a usable selector key and is never silently truncated)."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not float(number).is_integer():
        return None
    return int(number)


def addressable_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows whose season/event/session identity the selector can address: a whole-number
    season and non-blank event and session names."""
    season = pd.to_numeric(frame["season"], errors="coerce")
    whole = season.notna() & np.isfinite(season) & (season == np.floor(season))
    return whole & ~blank_mask(frame["event"]) & ~blank_mask(frame["session"])


def _chronological_order(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["season", "round_number", "event_date", "event", "session", "lap_number"] if c in frame.columns]


# Session codes as FastF1 writes them, plus the spelled-out forms a hand-made file may use.
SESSION_ALIASES = {
    "PRACTICE 1": "FP1", "PRACTICE 2": "FP2", "PRACTICE 3": "FP3", "QUALIFYING": "Q",
    "SPRINT QUALIFYING": "SQ", "SPRINT SHOOTOUT": "SS", "SPRINT": "S", "RACE": "R",
}
SPRINT_CODES = {"S", "SQ", "SS"}
# Running order of a weekend's sessions. Lexical order would put a Sprint ("S") after the
# Race ("R") and make it the "latest" session of a file; the race is the last session run.
# Sprint weekends have had three formats: 2021-2022 ran qualifying on Friday before FP2
# and the sprint (FP1, Q, FP2, S, R); 2023 replaced FP2 with the sprint shootout
# (FP1, Q, SS, S, R); from 2024 sprint qualifying and the sprint precede qualifying
# (FP1, SQ, S, Q, R). A conventional weekend is FP1, FP2, FP3, Q, R. Codes not in the
# table sort just before the race so the race stays the latest session when present.
_CONVENTIONAL = {"FP1": 0, "FP2": 1, "FP3": 2, "Q": 3, "R": 6}
_SPRINT_2021 = {"FP1": 0, "Q": 1, "FP2": 2, "S": 4, "R": 6}
_SPRINT_2023 = {"FP1": 0, "Q": 1, "SS": 2, "S": 4, "R": 6}
_SPRINT_2024 = {"FP1": 0, "SQ": 1, "S": 2, "Q": 3, "R": 6}
_UNKNOWN_RANK = 5
_MISSING_SEASON = "(missing season)"
# Kept for callers that only need the modern order (sprint weekends since 2024).
SESSION_ORDER = dict(_SPRINT_2024, FP2=1, FP3=2, SS=1)


def session_code(value: object) -> str:
    """Normalised session code: 'Sprint' and 'sprint ' become 'S', 'r' becomes 'R'."""
    code = str(value).strip().upper()
    return SESSION_ALIASES.get(code, code)


def _weekend_order(season: float, sprint_weekend: bool) -> dict[str, int]:
    if not sprint_weekend:
        return _CONVENTIONAL
    if pd.isna(season) or season >= 2024:
        return _SPRINT_2024
    if season >= 2023:
        return _SPRINT_2023
    return _SPRINT_2021


def session_rank(frame: pd.DataFrame) -> pd.Series:
    """Position of each row's session within its weekend (see the tables above). The weekend
    format is inferred per season/event: an event with any sprint session uses the sprint
    order of its season, every other event the conventional order."""
    codes = frame["session"].map(session_code)
    season = pd.to_numeric(frame["season"], errors="coerce") if "season" in frame.columns else pd.Series(np.nan, index=frame.index)
    # A missing season is keyed by a sentinel no numeric value can equal, so a file that
    # really contains season -1 (or any other number) is never mistaken for "unknown".
    season_key = season.astype(object).where(season.notna(), _MISSING_SEASON)
    group_keys = [season_key] + ([frame["event"].astype(str)] if "event" in frame.columns else [])
    sprint_weekend = codes.isin(SPRINT_CODES).groupby(group_keys).transform("any")
    # One rank per distinct (code, season, weekend format), then a single vectorised lookup,
    # so a multi-season import costs one pass however many combinations it holds.
    combos = pd.DataFrame({"code": codes, "season": season_key, "sprint": sprint_weekend}).drop_duplicates()
    combos["rank"] = [
        float(_weekend_order(np.nan if s == _MISSING_SEASON else float(s), bool(sp)).get(c, _UNKNOWN_RANK))
        for c, s, sp in zip(combos["code"], combos["season"], combos["sprint"])
    ]
    lookup = combos.set_index(["code", "season", "sprint"])["rank"]
    index = pd.MultiIndex.from_arrays([codes, season_key, sprint_weekend])
    return pd.Series(lookup.reindex(index).to_numpy(), index=frame.index, dtype=float)


def sort_chronologically(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in the order the laps were driven: season, round or date, event, session (by the
    weekend's running order, not alphabetically), lap number."""
    keys = _chronological_order(frame)
    if "session" not in keys:
        return frame.sort_values(keys, kind="stable")
    # The rank lives in a scratch column whose name cannot collide with user data; it is
    # dropped again so the caller gets its own columns back untouched.
    scratch = "_raceshift_session_rank"
    while scratch in frame.columns:
        scratch += "_"
    ranked = frame.assign(**{scratch: session_rank(frame)})
    keys = [scratch if k == "session" else k for k in keys]
    return ranked.sort_values(keys, kind="stable").drop(columns=scratch)


def inference_table_for(raw: pd.DataFrame, history: int = 5) -> pd.DataFrame:
    """Feature table for forecasting and backtesting, built with exactly the lap-validity rules
    used in training. The *whole* raw frame goes in, untimed laps included: the red-flag
    restart rule needs the (usually untimed) stoppage laps as evidence, so dropping rows
    without a lap time first would let restart laps through as "clean" laps with 30-60 s
    errors. Rows are every valid lap; the target is present only where a usable next lap exists."""
    return build_full_context_table(raw, history=int(history), require_target=False)


class RaceShiftArtifact:
    """A saved RaceShift FFR model plus its train-only preprocessor and feature contract."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        missing = missing_artifact_files(self.directory)
        if missing:
            raise FileNotFoundError(f"Artifact {self.directory.name} is incomplete, missing: {missing}")
        self.model = ForwardForwardRegressor.load(self.directory)
        self.contract = json.loads((self.directory / "feature_contract.json").read_text())
        self.metrics = json.loads((self.directory / "metrics.json").read_text())
        self.preprocessor = self._load_preprocessor()

    def _load_preprocessor(self):
        """Prefer the pickle-free JSON preprocessor spec written by the exporter (exact NumPy
        re-implementation, verified against sklearn at export time); fall back to the joblib
        pickle, silencing sklearn's version-mismatch warning, which is harmless for a fitted
        OneHotEncoder/StandardScaler and would otherwise print a dozen lines per load."""
        specs = sorted((self.directory / "export").glob("*_preprocessor.json")) if (self.directory / "export").is_dir() else []
        if specs:
            from raceshift.models.export import JsonPreprocessor

            return JsonPreprocessor(json.loads(specs[0].read_text()))
        import warnings

        with warnings.catch_warnings():
            try:
                from sklearn.exceptions import InconsistentVersionWarning

                warnings.simplefilter("ignore", InconsistentVersionWarning)
            except ImportError:  # pragma: no cover
                pass
            return joblib.load(self.directory / "preprocessor.joblib")

    @property
    def is_synthetic(self) -> bool:
        return bool(self.metrics.get("is_synthetic", False))

    @property
    def data_source(self) -> str:
        return str(self.metrics.get("data_source", "unknown"))

    @staticmethod
    def latest_session(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
        """Return the rows of the chronologically latest season/event/session and its key.
        Rows with a blank or fractional identity (listed as placeholders by ``list_sessions``)
        are never the default: the API serialises the key as an integer season and the
        selector could not address such a session anyway."""
        usable = raw[addressable_mask(raw)]
        if usable.empty:
            raise ValueError("No session with a whole-number season and non-blank event and session names in this file")
        ordered = sort_chronologically(usable)
        last = ordered.iloc[-1]
        key = {"season": last["season"], "event": last["event"], "session": last["session"]}
        mask = (
            (raw["season"] == key["season"]) & (raw["event"] == key["event"]) & (raw["session"] == key["session"])
        )
        return raw[mask], key

    @staticmethod
    def select_session(
        raw: pd.DataFrame, season: int | None = None, event: str | None = None, session: str | None = None
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Rows of one session in the file: the chronologically latest one by default, or the
        one named by ``event`` (plus ``season``/``session`` when the file has several)."""
        if event is None and season is None and session is None:
            return RaceShiftArtifact.latest_session(raw)
        mask = pd.Series(True, index=raw.index)
        if event is not None:
            mask &= raw["event"].astype(str) == str(event)
        if season is not None:
            mask &= pd.to_numeric(raw["season"], errors="coerce") == int(season)
        if session is not None:
            mask &= raw["session"].astype(str) == str(session)
        scope = raw[mask]
        if scope.empty:
            wanted = " ".join(str(v) for v in (season, event, session) if v is not None)
            raise ValueError(f"No laps for session {wanted!r} in this file")
        return RaceShiftArtifact.latest_session(scope)

    @staticmethod
    def list_sessions(raw: pd.DataFrame) -> list[dict[str, object]]:
        """Every season/event/session in the file in chronological order with lap and driver counts."""
        cols = ["season", "event", "session"]
        ordered = sort_chronologically(raw)
        out = []
        # Rows with a null season/event/session cannot be addressed by the selector; they are
        # listed under an explicit placeholder so nothing disappears silently (the summary also
        # reports them as a data warning).
        for (season, event, session), group in ordered.groupby(cols, sort=False, dropna=False):
            season_number = integer_season(season)
            selectable = season_number is not None and not is_blank(event) and not is_blank(session)
            out.append({
                "season": season_number,
                "event": str(event) if not is_blank(event) else "(missing event)",
                "session": str(session) if not is_blank(session) else "(missing session)",
                # False for the placeholder rows: the selector cannot address them, so the UI
                # lists them disabled instead of sending the placeholder back as a filter.
                "selectable": bool(selectable),
                "laps": int(len(group)),
                "drivers": int(group["driver"].astype(str).nunique()),
                "driver_codes": sorted(group["driver"].astype(str).unique().tolist()),
                "date": str(group["event_date"].iloc[0])[:10] if "event_date" in group and pd.notna(group["event_date"].iloc[0]) else None,
            })
        return out


    @staticmethod
    def _pick_driver(scope: pd.DataFrame, driver: str | None) -> tuple[str, list[str]]:
        """Default driver: most completed laps, then best position on the last lap (the race
        winner when the whole field finished), then code order. Explicit choices are validated."""
        available = sorted(scope["driver"].unique().tolist())
        if driver is not None:
            driver = str(driver)
            if driver not in available:
                raise ValueError(f"Driver {driver!r} not found in the latest session. Available: {available}")
            return driver, available
        last = scope.sort_values("lap_number").groupby("driver").tail(1)
        position = pd.to_numeric(last.get("position"), errors="coerce") if "position" in last else pd.Series(np.nan, index=last.index)
        ranked = pd.DataFrame({"driver": last["driver"].to_numpy(), "laps": last["lap_number"].to_numpy(dtype=float), "position": position.fillna(99).to_numpy()})
        ranked = ranked.sort_values(["laps", "position", "driver"], ascending=[False, True, True], kind="stable")
        return str(ranked.iloc[0]["driver"]), available

    @property
    def history_laps(self) -> int:
        return int(self.contract.get("history", 5))

    def inference_table(self, raw: pd.DataFrame) -> pd.DataFrame:
        """The leakage-safe feature table for inference: every valid lap, targets only where a
        usable next lap exists. The API caches this per file so repeated calls are cheap."""
        return inference_table_for(raw, history=self.history_laps)

    def forecast_last_available(
        self,
        raw: pd.DataFrame,
        driver: str | None = None,
        table: pd.DataFrame | None = None,
        season: int | None = None,
        event: str | None = None,
        session: str | None = None,
    ) -> dict:
        """Forecast the next lap for the latest completed lap in the supplied history.

        Scope is the chronologically latest session in the file. When `driver` is not given,
        the driver who finished best with the most completed laps is used. Historical priors
        still see the whole file because earlier events feed them. ``table`` may be a
        precomputed :meth:`inference_table` for the same ``raw`` frame.
        """
        missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        data = raw[pd.to_numeric(raw["lap_time_s"], errors="coerce").notna()]
        if data.empty:
            raise ValueError("No rows with a lap time available for forecast")

        scope, key = self.select_session(data, season=season, event=event, session=session)
        scope = scope.copy()
        scope["driver"] = scope["driver"].astype(str)
        driver, available_drivers = self._pick_driver(scope, driver)

        history = scope[scope["driver"] == driver].sort_values("lap_number")
        if len(history) < 2:
            raise ValueError("Need at least two completed laps for the selected driver to forecast the next one")
        latest = history.iloc[-1].copy()

        history_laps = self.history_laps
        table = self.inference_table(raw) if table is None else table
        candidates = table[
            (table["season"] == key["season"])
            & (table["event"] == key["event"])
            & (table["session"] == key["session"])
            & (table["driver"].astype(str) == driver)
        ].sort_values("lap_number")
        if candidates.empty or int(candidates.iloc[-1]["lap_number"]) != int(latest["lap_number"]):
            raise ValueError(
                "The latest lap for this driver is not a usable completed lap (pit, flagged, restart, deleted or inaccurate). "
                "Import more completed laps or choose another driver."
            )
        row = candidates.iloc[-1:]

        features = self.contract["numeric"] + self.contract["categorical"]
        x = np.asarray(self.preprocessor.transform(row[features]), dtype=np.float32)
        pred = self.model.predict_with_uncertainty(x)
        baseline = float(row["rolling_median_5"].iloc[0])
        predicted = baseline + float(pred["prediction"][0])
        if not np.isfinite(predicted):
            raise ValueError("Model produced a non-finite forecast for this row")

        completed = int(len(history))
        # The forecast is for the lap after the last recorded one. When that lap is the last
        # lap anybody completed in the session, the race is over and the forecast is hypothetical.
        session_last_lap = int(pd.to_numeric(scope["lap_number"], errors="coerce").max())
        race_finished = int(latest["lap_number"]) >= session_last_lap

        def _num(col: str):
            if col not in row.columns:
                return None
            value = row[col].iloc[0]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def _text(col: str):
            if col not in row.columns:
                return None
            value = row[col].iloc[0]
            return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

        context = {
            "compound": _text("compound"),
            "tyre_life": _num("tyre_life"),
            "stint": _num("stint"),
            "position": _num("position"),
            "laps_in_segment": _num("laps_in_segment"),
            "track_temp_c": _num("track_temp_c"),
            "air_temp_c": _num("air_temp_c"),
            "humidity_pct": _num("humidity_pct"),
            "wind_speed_ms": _num("wind_speed_ms"),
            "wind_direction_deg": _num("wind_direction_deg"),
            "rainfall": _text("rainfall"),
            "gap_ahead_s": _num("gap_ahead_s"),
            "gap_behind_s": _num("gap_behind_s"),
            "team": _text("team"),
            "circuit": _text("circuit"),
        }
        historical_context = {
            "driver_circuit_pace_s": _num("hist_driver_circuit_pace"),
            "team_circuit_pace_s": _num("hist_team_circuit_pace"),
            "compound_circuit_pace_s": _num("hist_compound_circuit_pace"),
            "driver_circuit_compound_pace_s": _num("hist_driver_circuit_compound_pace"),
            "matched_weather_compound_pace_s": _num("hist_weather_compound_pace"),
            "driver_matched_weather_pace_s": _num("hist_driver_weather_pace"),
            "driver_overall_pace_s": _num("hist_driver_global_pace"),
            "team_overall_pace_s": _num("hist_team_global_pace"),
            "note": "Medians of earlier events only. None means no earlier event matched.",
        }
        return {
            "context": context,
            "historical_context": historical_context,
            "season": int(key["season"]),
            "event": str(key["event"]),
            "session": str(key["session"]),
            "driver": driver,
            "available_drivers": available_drivers,
            "lap_number_completed": int(row["lap_number"].iloc[0]),
            "completed_laps_in_session": completed,
            "history_laps_used": history_laps,
            "short_history": completed < history_laps,
            "race_finished": bool(race_finished),
            "session_last_lap": session_last_lap,
            "predicted_next_lap_s": predicted,
            "lower_80_s": baseline + float(pred["lower_80"][0]),
            "upper_80_s": baseline + float(pred["upper_80"][0]),
            "layer_disagreement_s": float(pred["layer_disagreement"][0]),
            "rolling5_baseline_s": baseline,
            "last_lap_time_s": float(latest["lap_time_s"]),
            "artifact": self.directory.name,
            "data_source": self.data_source,
            "is_synthetic": self.is_synthetic,
        }

    def backtest_session(
        self,
        raw: pd.DataFrame,
        driver: str | None = None,
        laps: int = 10,
        table: pd.DataFrame | None = None,
        season: int | None = None,
        event: str | None = None,
        session: str | None = None,
    ) -> dict:
        """Score the model on the last ``laps`` completed lap pairs of one driver in the latest session.

        Every row is a lap N whose next lap N+1 was actually driven, so predicted and actual
        can be compared. Only pairs where both laps are valid racing laps count (pit, yellow-flag,
        safety-car, red-flag, restart and deleted laps are skipped, exactly as in training). Nothing here looks past
        lap N when predicting lap N+1: the feature table is the same leakage-safe table used for
        training and the model never sees the actual next lap.
        """
        if laps < 1:
            raise ValueError("laps must be >= 1")
        missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        data = raw.copy()
        data = data[pd.to_numeric(data["lap_time_s"], errors="coerce").notna()]
        if data.empty:
            raise ValueError("No rows with a lap time available for a backtest")

        scope, key = self.select_session(data, season=season, event=event, session=session)
        scope = scope.copy()
        scope["driver"] = scope["driver"].astype(str)
        driver, available_drivers = self._pick_driver(scope, driver)
        driven = int((scope["driver"] == driver).sum())

        table = self.inference_table(raw) if table is None else table
        rows = table[
            (table["season"] == key["season"])
            & (table["event"] == key["event"])
            & (table["session"] == key["session"])
            & (table["driver"].astype(str) == driver)
        ].sort_values("lap_number")
        rows = rows[pd.to_numeric(rows[RAW_TARGET_COLUMN], errors="coerce").notna()]
        usable = int(len(rows))
        if rows.empty:
            raise ValueError(
                "No completed lap pairs for this driver where both laps are valid racing laps "
                "(pit, yellow-flag, safety-car, red-flag, restart and deleted laps are excluded, as in training)."
            )
        rows = rows.tail(int(laps))

        features = self.contract["numeric"] + self.contract["categorical"]
        x = np.asarray(self.preprocessor.transform(rows[features]), dtype=np.float32)
        pred = self.model.predict_with_uncertainty(x)
        baseline = rows["rolling_median_5"].to_numpy(dtype=float)
        predicted = baseline + np.asarray(pred["prediction"], dtype=float)
        lower = baseline + np.asarray(pred["lower_80"], dtype=float)
        upper = baseline + np.asarray(pred["upper_80"], dtype=float)
        actual = rows[RAW_TARGET_COLUMN].to_numpy(dtype=float)
        current = rows["lap_time_s"].to_numpy(dtype=float)
        error = predicted - actual
        abs_error = np.abs(error)
        inside = (actual >= lower) & (actual <= upper)

        def _opt(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        lap_rows = []
        for i in range(len(rows)):
            r = rows.iloc[i]
            lap_rows.append(
                {
                    "lap_number_completed": int(r["lap_number"]),
                    "next_lap_number": int(r["lap_number"]) + 1,
                    "actual_next_lap_s": float(actual[i]),
                    "predicted_next_lap_s": float(predicted[i]),
                    "lower_80_s": float(lower[i]),
                    "upper_80_s": float(upper[i]),
                    "rolling5_baseline_s": float(baseline[i]),
                    "previous_lap_s": float(current[i]),
                    "error_s": float(error[i]),
                    "abs_error_s": float(abs_error[i]),
                    "within_interval": bool(inside[i]),
                    "compound": None if pd.isna(r.get("compound")) else str(r.get("compound")),
                    "tyre_life": _opt(r.get("tyre_life")),
                    "position": _opt(r.get("position")),
                }
            )
        summary = {
            "rows": int(len(rows)),
            "mae_s": float(abs_error.mean()),
            "rmse_s": float(np.sqrt(np.mean(error**2))),
            "p90_ae_s": float(np.quantile(abs_error, 0.9)),
            "signed_bias_s": float(error.mean()),
            "within_0_5s_share": float((abs_error <= 0.5).mean()),
            "within_1s_share": float((abs_error <= 1.0).mean()),
            "interval80_coverage": float(inside.mean()),
            "previous_lap_mae_s": float(np.abs(current - actual).mean()),
            "rolling5_mae_s": float(np.abs(baseline - actual).mean()),
        }
        return {
            "season": int(key["season"]),
            "event": str(key["event"]),
            "session": str(key["session"]),
            "driver": driver,
            "available_drivers": available_drivers,
            "laps_driven": driven,
            "usable_lap_pairs": usable,
            "skipped_laps": max(driven - usable, 0),
            "laps": lap_rows,
            "summary": summary,
            "artifact": self.directory.name,
            "data_source": self.data_source,
            "is_synthetic": self.is_synthetic,
        }
