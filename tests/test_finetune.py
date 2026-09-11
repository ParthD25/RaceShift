"""Forward-forward fine-tuning: local updates only, readout refit, and the round-based split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raceshift.data.splits import season_round_split
from raceshift.models.forward_forward_regressor import FFRConfig, ForwardForwardRegressor
from raceshift.train.finetune import parse_rounds


def _data(rng, n, shift=0.0):
    x = rng.normal(size=(n, 10)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] + shift * x[:, 2]).astype(np.float32)
    return x, y


def test_continue_fit_adapts_with_local_updates_only():
    rng = np.random.default_rng(3)
    x, y = _data(rng, 3000)
    x_new, y_new = _data(rng, 1200, shift=0.8)
    x_test, y_test = _data(rng, 800, shift=0.8)
    cfg = FFRConfig(layer_nodes=(32, 32), ordinal_groups=(8, 16), epochs_per_layer=12, batch_size=128)
    model = ForwardForwardRegressor(10, cfg).fit(x, y)
    before = float(np.abs(model.predict(x_test) - y_test).mean())
    center, scale = model.target_center, model.target_scale

    # A local step on layer 1 never touches layer 2 (no gradient flows across layers).
    layer2 = model.layers[1].weight.copy()
    model.layers[0].local_train_step(x_new[:128], model._scale_target(y_new[:128]))
    assert np.array_equal(layer2, model.layers[1].weight)

    model.continue_fit(x_new, y_new, epochs_per_layer=8, validation=(x_test, y_test))
    after = float(np.abs(model.predict(x_test) - y_test).mean())
    assert after < before
    assert (model.target_center, model.target_scale) == (center, scale)  # scaling is kept
    assert any(h.get("phase") == "finetune" for h in model.training_history)
    assert model.interval_q80 > 0


def test_continue_fit_requires_a_fitted_model_and_matching_width():
    cfg = FFRConfig(layer_nodes=(32, 32), ordinal_groups=(8, 16), epochs_per_layer=1, batch_size=64)
    model = ForwardForwardRegressor(10, cfg)
    with pytest.raises(RuntimeError):
        model.continue_fit(np.zeros((4, 10), np.float32), np.zeros(4, np.float32))
    rng = np.random.default_rng(0)
    x, y = _data(rng, 300)
    model.fit(x, y)
    with pytest.raises(ValueError):
        model.continue_fit(np.zeros((4, 9), np.float32), np.zeros(4, np.float32))


def test_parse_rounds():
    assert parse_rounds("1-5") == (1, 5) and parse_rounds("8-") == (8, 99) and parse_rounds("6") == (6, 6) and parse_rounds(None) is None


def test_season_round_split_can_train_through_early_rounds():
    frame = pd.DataFrame({
        "season": [2024] * 4 + [2026] * 8,
        "round_number": [1, 2, 3, 4] + list(range(1, 9)),
        "lap_number": 1,
    })
    train, val, test = season_round_split(frame, 2024, 2026, cutoff_round=5, train_through_round=3)
    assert sorted(train[train.season == 2026].round_number) == [1, 2, 3]
    assert sorted(val.round_number) == [4, 5] and sorted(test.round_number) == [6, 7, 8]
    with pytest.raises(ValueError):
        season_round_split(frame, 2024, 2026, cutoff_round=5, train_through_round=5)
