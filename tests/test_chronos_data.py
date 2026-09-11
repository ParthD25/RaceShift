from __future__ import annotations

import numpy as np

from make_synthetic_fixture import build_fixture
from raceshift.features.lap_features import build_next_lap_table
from raceshift.foundation.chronos_data import example_to_chronos_input, make_walk_forward_examples


def test_walk_forward_examples_never_include_the_target_lap():
    raw = build_fixture(seasons=(2024,), events_per_season=1, drivers=("AAA", "BBB"), laps=12)
    table = build_next_lap_table(raw)
    examples = make_walk_forward_examples(table, context_length=6, stride=1, min_context=4)
    assert examples, "expected at least one example"
    by_driver = raw.set_index(["driver", "lap_number"])["lap_time_s"]
    for ex in examples:
        assert len(ex.context) <= 6
        assert len(ex.context) >= 4
        # Context ends at lap N and the answer is the real lap N+1 from the raw table.
        assert ex.context[-1] == np.float32(by_driver[(ex.driver, ex.lap_number)])
        assert ex.actual_next == np.float32(by_driver[(ex.driver, ex.lap_number + 1)])
        assert example_to_chronos_input(ex).dtype == np.float32


def test_walk_forward_stride_reduces_examples():
    raw = build_fixture(seasons=(2024,), events_per_season=1, drivers=("AAA",), laps=20)
    table = build_next_lap_table(raw)
    dense = make_walk_forward_examples(table, context_length=5, stride=1)
    sparse = make_walk_forward_examples(table, context_length=5, stride=3)
    assert len(sparse) < len(dense)
