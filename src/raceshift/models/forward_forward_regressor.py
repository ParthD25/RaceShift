from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class FFRConfig:
    """Deep Forward-Forward regression configuration.

    Each layer is locally trained. There is no global backward pass and no autograd.
    The production Colab default intentionally contains multiple deep hidden layers.
    """

    layer_nodes: tuple[int, ...] = (512, 384, 256, 192)
    ordinal_groups: tuple[int, ...] = (8, 16, 32, 64)
    epochs_per_layer: int = 80
    batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    temperature: float = 0.55
    target_sigma_groups: float = 1.1
    ridge_alpha: float = 2.0
    seed: int = 42
    clip_update_norm: float = 5.0

    def validate(self) -> None:
        if len(self.layer_nodes) != len(self.ordinal_groups):
            raise ValueError("layer_nodes and ordinal_groups must have the same length")
        if len(self.layer_nodes) < 2:
            raise ValueError("RaceShift FFR requires at least two locally trained layers")
        for nodes, groups in zip(self.layer_nodes, self.ordinal_groups):
            if nodes < groups:
                raise ValueError("Each layer must have at least as many nodes as ordinal groups")
            if nodes % groups != 0:
                raise ValueError("Each layer's node count must be divisible by its ordinal group count")


class _LocalAdam:
    """Adam applied only to the current layer's analytically derived local update."""

    def __init__(self, shape: tuple[int, ...], lr: float, weight_decay: float):
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        grad = grad.astype(np.float32, copy=False) + self.weight_decay * param
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * (grad * grad)
        m_hat = self.m / (1.0 - 0.9**self.t)
        v_hat = self.v / (1.0 - 0.999**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class _FFLocalLayer:
    def __init__(self, in_dim: int, nodes: int, groups: int, config: FFRConfig, rng: np.random.Generator):
        self.in_dim = int(in_dim)
        self.nodes = int(nodes)
        self.groups = int(groups)
        self.nodes_per_group = self.nodes // self.groups
        scale = np.sqrt(2.0 / max(1, self.in_dim))
        self.weight = rng.normal(0.0, scale, size=(self.in_dim, self.nodes)).astype(np.float32)
        self.bias = np.zeros(self.nodes, dtype=np.float32)
        self._adam_w = _LocalAdam(self.weight.shape, config.learning_rate, config.weight_decay)
        self._adam_b = _LocalAdam(self.bias.shape, config.learning_rate, 0.0)
        self.temperature = float(config.temperature)
        self.target_sigma_groups = float(config.target_sigma_groups)
        self.clip_update_norm = float(config.clip_update_norm)

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-6)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_norm = self._normalize_rows(x.astype(np.float32, copy=False))
        z = x_norm @ self.weight + self.bias
        return np.maximum(z, 0.0).astype(np.float32)

    def goodness(self, hidden: np.ndarray) -> np.ndarray:
        shaped = hidden.reshape(hidden.shape[0], self.groups, self.nodes_per_group)
        return np.mean(shaped * shaped, axis=2)

    def _soft_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        centers = (y_scaled + 1.0) * 0.5 * (self.groups - 1)
        idx = np.arange(self.groups, dtype=np.float32)[None, :]
        dist2 = (idx - centers[:, None]) ** 2
        target = np.exp(-0.5 * dist2 / max(self.target_sigma_groups**2, 1e-6))
        return target / np.maximum(target.sum(axis=1, keepdims=True), 1e-8)

    def local_loss(self, x: np.ndarray, y_scaled: np.ndarray) -> float:
        """The layer's own objective: cross entropy between soft ordinal targets and the
        softmax over group goodness. Depends on this layer's weights only."""
        return self.local_gradient(x, y_scaled)[0]

    def local_gradient(self, x: np.ndarray, y_scaled: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Loss and its analytic gradient with respect to this layer's weight and bias.

        The computation uses only this layer's input, weights and targets; there is no
        reference to any other layer, so nothing can flow backwards. Arithmetic follows the
        dtype of ``self.weight`` (float32 in training; tests use float64 to compare against
        finite differences).
        """
        dtype = self.weight.dtype
        x_norm = self._normalize_rows(x.astype(dtype, copy=False))
        z = x_norm @ self.weight + self.bias
        hidden = np.maximum(z, 0.0)
        goodness = self.goodness(hidden)
        target = self._soft_targets(y_scaled.astype(dtype, copy=False))

        logits = goodness / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-8)
        loss = -np.mean(np.sum(target * np.log(np.maximum(probs, 1e-8)), axis=1))

        # Local analytic derivative of cross entropy with respect to group goodness.
        batch = max(1, x.shape[0])
        d_goodness = (probs - target) / (batch * self.temperature)
        d_hidden = np.repeat(d_goodness[:, :, None], self.nodes_per_group, axis=2)
        d_hidden = d_hidden.reshape(hidden.shape)
        d_hidden *= (2.0 * hidden / self.nodes_per_group)
        d_z = d_hidden * (z > 0.0)
        d_w = x_norm.T @ d_z
        d_b = d_z.sum(axis=0)
        return float(loss), d_w, d_b

    def local_train_step(self, x: np.ndarray, y_scaled: np.ndarray) -> float:
        """One local update with a manually derived gradient.

        This is not global backpropagation. No gradient is propagated through other
        layers and no autograd engine is invoked.
        """
        loss, d_w, d_b = self.local_gradient(x, y_scaled)

        total_norm = float(np.sqrt(np.sum(d_w * d_w) + np.sum(d_b * d_b)))
        if total_norm > self.clip_update_norm:
            scale = self.clip_update_norm / max(total_norm, 1e-8)
            d_w *= scale
            d_b *= scale

        self.weight = self._adam_w.step(self.weight, d_w)
        self.bias = self._adam_b.step(self.bias, d_b)
        return float(loss)

    def local_prediction(self, hidden: np.ndarray) -> np.ndarray:
        goodness = self.goodness(hidden)
        weights = np.exp(goodness - goodness.max(axis=1, keepdims=True))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        centers = np.linspace(-1.0, 1.0, self.groups, dtype=np.float32)
        return (weights * centers[None, :]).sum(axis=1)


class ForwardForwardRegressor:
    """Multi-layer, multi-node Forward-Forward regressor for RaceShift.

    Layers are trained sequentially with independent local objectives. Hidden outputs
    are detached by construction because all training is explicit NumPy computation.
    The final numeric readout is solved in closed form with ridge regression.
    """

    def __init__(self, input_dim: int, config: FFRConfig | None = None):
        self.config = config or FFRConfig()
        self.config.validate()
        self.input_dim = int(input_dim)
        self.layers: list[_FFLocalLayer] = []
        self.target_center = 0.0
        self.target_scale = 1.0
        self.readout_coef: np.ndarray | None = None
        self.interval_q80 = 1.0
        self.training_history: list[dict[str, float | int]] = []
        rng = np.random.default_rng(self.config.seed)
        dim = self.input_dim
        for nodes, groups in zip(self.config.layer_nodes, self.config.ordinal_groups):
            self.layers.append(_FFLocalLayer(dim, nodes, groups, self.config, rng))
            dim = nodes

    def _scale_target(self, y: np.ndarray) -> np.ndarray:
        return np.clip((y - self.target_center) / self.target_scale, -1.0, 1.0).astype(np.float32)

    def _unscale_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.target_scale + self.target_center

    def _represent(self, x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        current = x.astype(np.float32, copy=False)
        blocks: list[np.ndarray] = []
        for layer in self.layers:
            hidden = layer.forward(current)
            good = layer.goodness(hidden)
            local_pred = layer.local_prediction(hidden)[:, None]
            blocks.append(np.concatenate([good, local_pred], axis=1))
            current = _FFLocalLayer._normalize_rows(hidden)
        return blocks, np.concatenate(blocks, axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, validation: tuple[np.ndarray, np.ndarray] | None = None) -> "ForwardForwardRegressor":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be 2D and aligned with y")

        self.target_center = float(np.nanmedian(y))
        lo, hi = np.nanquantile(y, [0.02, 0.98])
        self.target_scale = float(max(abs(lo - self.target_center), abs(hi - self.target_center), 0.25))
        y_scaled = self._scale_target(y)

        rng = np.random.default_rng(self.config.seed + 17)
        current = x
        self.training_history = []
        for layer_idx, layer in enumerate(self.layers):
            for epoch in range(self.config.epochs_per_layer):
                order = rng.permutation(len(current))
                losses = []
                for start in range(0, len(order), self.config.batch_size):
                    idx = order[start : start + self.config.batch_size]
                    losses.append(layer.local_train_step(current[idx], y_scaled[idx]))
                self.training_history.append({
                    "layer": layer_idx + 1,
                    "epoch": epoch + 1,
                    "local_loss": float(np.mean(losses) if losses else np.nan),
                })
            current = _FFLocalLayer._normalize_rows(layer.forward(current))

        _, features = self._represent(x)
        design = np.concatenate([np.ones((len(features), 1), dtype=np.float32), features], axis=1).astype(np.float64)
        target = y.astype(np.float64)
        reg = self.config.ridge_alpha * np.eye(design.shape[1], dtype=np.float64)
        reg[0, 0] = 0.0
        self.readout_coef = np.linalg.solve(design.T @ design + reg, design.T @ target).astype(np.float32)

        if validation is not None:
            vx, vy = validation
            residual = np.abs(np.asarray(vy, dtype=np.float32).reshape(-1) - self.predict(np.asarray(vx, dtype=np.float32)))
            if len(residual):
                self.interval_q80 = float(np.nanquantile(residual, 0.80))
        else:
            residual = np.abs(y - self.predict(x))
            self.interval_q80 = float(np.nanquantile(residual, 0.80)) if len(residual) else 1.0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.readout_coef is None:
            raise RuntimeError("Model must be fitted before prediction")
        _, features = self._represent(np.asarray(x, dtype=np.float32))
        design = np.concatenate([np.ones((len(features), 1), dtype=np.float32), features], axis=1)
        return (design @ self.readout_coef).astype(np.float32)

    def predict_with_uncertainty(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        blocks, _ = self._represent(x)
        prediction = self.predict(x)
        layer_scaled = np.stack([b[:, -1] for b in blocks], axis=1)
        layer_predictions = self._unscale_target(layer_scaled)
        disagreement = np.std(layer_predictions, axis=1)
        half_width = np.maximum(self.interval_q80, disagreement).astype(np.float32)
        return {
            "prediction": prediction,
            "lower_80": prediction - half_width,
            "upper_80": prediction + half_width,
            "layer_disagreement": disagreement.astype(np.float32),
        }

    def architecture_summary(self) -> list[dict[str, int]]:
        return [
            {"layer": i + 1, "input_nodes": layer.in_dim, "hidden_nodes": layer.nodes, "ordinal_groups": layer.groups}
            for i, layer in enumerate(self.layers)
        ]

    def save(self, directory: str | Path) -> None:
        if self.readout_coef is None:
            raise RuntimeError("Cannot save an unfitted model")
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"readout_coef": self.readout_coef}
        for i, layer in enumerate(self.layers):
            arrays[f"layer_{i}_weight"] = layer.weight
            arrays[f"layer_{i}_bias"] = layer.bias
        np.savez_compressed(out / "model_weights.npz", **arrays)
        metadata = {
            "model_type": "RaceShiftFFR",
            "input_dim": self.input_dim,
            "config": asdict(self.config),
            "target_center": self.target_center,
            "target_scale": self.target_scale,
            "interval_q80": self.interval_q80,
            "architecture": self.architecture_summary(),
            "training_policy": "local-forward-forward-no-global-backprop",
        }
        (out / "model_config.json").write_text(json.dumps(metadata, indent=2))
        (out / "training_history.json").write_text(json.dumps(self.training_history, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "ForwardForwardRegressor":
        path = Path(directory)
        metadata = json.loads((path / "model_config.json").read_text())
        cfg_data = metadata["config"]
        cfg_data["layer_nodes"] = tuple(cfg_data["layer_nodes"])
        cfg_data["ordinal_groups"] = tuple(cfg_data["ordinal_groups"])
        model = cls(metadata["input_dim"], FFRConfig(**cfg_data))
        weights = np.load(path / "model_weights.npz")
        for i, layer in enumerate(model.layers):
            layer.weight = weights[f"layer_{i}_weight"].astype(np.float32)
            layer.bias = weights[f"layer_{i}_bias"].astype(np.float32)
        model.readout_coef = weights["readout_coef"].astype(np.float32)
        model.target_center = float(metadata["target_center"])
        model.target_scale = float(metadata["target_scale"])
        model.interval_q80 = float(metadata["interval_q80"])
        history = path / "training_history.json"
        model.training_history = json.loads(history.read_text()) if history.exists() else []
        return model
