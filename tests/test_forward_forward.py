from __future__ import annotations

import inspect
import numpy as np

from raceshift.models.forward_forward_regressor import FFRConfig, ForwardForwardRegressor


def test_multilayer_multinode_architecture():
    cfg = FFRConfig(
        layer_nodes=(64, 48, 32, 24),
        ordinal_groups=(8, 8, 8, 8),
        epochs_per_layer=2,
        batch_size=32,
    )
    model = ForwardForwardRegressor(12, cfg)
    summary = model.architecture_summary()
    assert len(summary) == 4
    assert [x["hidden_nodes"] for x in summary] == [64, 48, 32, 24]
    assert summary[0]["input_nodes"] == 12
    assert summary[1]["input_nodes"] == 64


def test_training_has_no_backward_or_autograd_calls():
    import raceshift.models.forward_forward_regressor as module
    source = inspect.getsource(module)
    forbidden = [".backward(", "autograd.grad("]
    for token in forbidden:
        assert token not in source


def _layer64(seed: int = 0):
    from raceshift.models.forward_forward_regressor import _FFLocalLayer

    cfg = FFRConfig(layer_nodes=(12,), ordinal_groups=(4,), temperature=0.7, target_sigma_groups=1.0)
    layer = _FFLocalLayer(6, 12, 4, cfg, np.random.default_rng(seed))
    # float64 so finite differences are not dominated by float32 rounding
    layer.weight = layer.weight.astype(np.float64)
    layer.bias = np.random.default_rng(seed + 1).normal(0, 0.1, size=layer.bias.shape)
    return layer


def test_local_gradient_matches_finite_differences():
    """The hand-derived gradient in local_gradient is checked numerically, entry by entry."""
    layer = _layer64()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 6))
    y = rng.uniform(-1, 1, size=16)
    loss, d_w, d_b = layer.local_gradient(x, y)
    assert np.isfinite(loss) and d_w.shape == layer.weight.shape and d_b.shape == layer.bias.shape

    eps = 1e-6
    num_w = np.zeros_like(d_w)
    for i in range(layer.weight.shape[0]):
        for j in range(layer.weight.shape[1]):
            saved = layer.weight[i, j]
            layer.weight[i, j] = saved + eps
            up = layer.local_loss(x, y)
            layer.weight[i, j] = saved - eps
            down = layer.local_loss(x, y)
            layer.weight[i, j] = saved
            num_w[i, j] = (up - down) / (2 * eps)
    num_b = np.zeros_like(d_b)
    for j in range(layer.bias.shape[0]):
        saved = layer.bias[j]
        layer.bias[j] = saved + eps
        up = layer.local_loss(x, y)
        layer.bias[j] = saved - eps
        down = layer.local_loss(x, y)
        layer.bias[j] = saved
        num_b[j] = (up - down) / (2 * eps)

    assert np.max(np.abs(d_w)) > 1e-6, "gradient is degenerate; the test would pass trivially"
    np.testing.assert_allclose(d_w, num_w, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(d_b, num_b, rtol=1e-4, atol=1e-7)


def test_layer_update_is_independent_of_later_layers():
    """Nothing flows backwards: layer k's gradient and update are bit-identical whatever the
    weights of layers k+1... are, and only layer k's parameters change in its step."""
    cfg = FFRConfig(layer_nodes=(24, 16, 8), ordinal_groups=(4, 4, 4), epochs_per_layer=1, batch_size=32)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 9)).astype(np.float32)
    y = rng.uniform(-1, 1, size=64).astype(np.float32)

    a = ForwardForwardRegressor(9, cfg)
    b = ForwardForwardRegressor(9, cfg)
    for later in b.layers[1:]:
        later.weight = rng.normal(size=later.weight.shape).astype(np.float32)
        later.bias = rng.normal(size=later.bias.shape).astype(np.float32)

    loss_a, dw_a, db_a = a.layers[0].local_gradient(x, y)
    loss_b, dw_b, db_b = b.layers[0].local_gradient(x, y)
    assert loss_a == loss_b
    assert np.array_equal(dw_a, dw_b) and np.array_equal(db_a, db_b)

    before = [(l.weight.copy(), l.bias.copy()) for l in a.layers]
    a.layers[0].local_train_step(x, y)
    assert not np.array_equal(before[0][0], a.layers[0].weight)
    for (w0, b0), layer in zip(before[1:], a.layers[1:]):
        assert np.array_equal(w0, layer.weight) and np.array_equal(b0, layer.bias)


def test_forward_forward_fit_predict_smoke():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(180, 10)).astype(np.float32)
    y = (0.25 * x[:, 0] - 0.15 * x[:, 1] + 0.1 * x[:, 2] + rng.normal(0, 0.04, 180)).astype(np.float32)
    cfg = FFRConfig(
        layer_nodes=(48, 32, 24),
        ordinal_groups=(8, 8, 8),
        epochs_per_layer=3,
        batch_size=48,
        learning_rate=0.002,
    )
    model = ForwardForwardRegressor(x.shape[1], cfg).fit(x[:140], y[:140], validation=(x[140:160], y[140:160]))
    pred = model.predict(x[160:])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()
    interval = model.predict_with_uncertainty(x[160:])
    assert np.all(interval["upper_80"] >= interval["lower_80"])
