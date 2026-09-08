"""Tabular GAN trained with DP-SGD (Opacus), class-conditioned.

Classic DP-GAN construction: only the discriminator sees the data and trains
with DP-SGD (gradient clipping + noise). The generator is post-processing of
the discriminator, so the result is DP with the accounted epsilon.

For utility the AC-GAN scheme is used: the generator receives a condition
vector (class of the most imbalanced column) and the discriminator has an
auxiliary head that must predict it. The condition is sampled balanced in the
generator step (CTGAN-style) so minority classes do not collapse to the
majority one; when sampling, the empirical frequency is used to respect the
marginal. Numerics use mode-specific normalization (``ModeEncoder``) to avoid
flattening modes.

The RDP accountant translates noise, epochs and sample size into the real
accumulated epsilon, exposed in ``accounted_epsilon`` after ``fit``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from synthpriv.core.base import BaseSynthesizer
from synthpriv.core.registry import register_generator
from synthpriv.dp.encoder import ModeEncoder
from synthpriv.privacy.assurance import DpAssurance, assert_dp
from synthpriv.privacy.mechanisms import DPSGD
from synthpriv.utils import get_logger

# Benign and abundant Opacus/Torch warnings during DP training.
_BN_WARNINGS = ("Full backward hook", "Secure RNG turned off", "Optimal order is the largest alpha")
for _m in _BN_WARNINGS:
    warnings.filterwarnings("ignore", message=f"{_m}.*")

logger = get_logger("dp")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Generator(nn.Module):
    """Conditional MLP: ``(z, cond)`` -> vector in the ``ModeEncoder`` space.

    The output is assembled in blocks: each numeric contributes ``tanh(value)`` +
    a softmax over its modes, and each categorical its softmax. The result is
    isomorphic to the encoding.
    """

    def __init__(self, latent, hidden, encoder: ModeEncoder, n_layers=2):
        super().__init__()
        self.latent = latent
        self.encoder = encoder
        self.blocks = encoder.blocks
        self.clip = encoder.clip_value
        self.cond_block = next(
            (b for b in self.blocks if b["type"] == "cat" and b["col"] == encoder.condition_column_),
            None)
        in_dim = latent + encoder.n_cond
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, encoder.total_dims)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, cond=None):
        if cond is not None:
            inp = torch.cat([z, cond], dim=1)
        else:
            inp = z
        raw = self.net(inp)
        outs = []
        for b in self.blocks:
            if b["type"] == "num":
                if b.get("kind") == "uniform":
                    outs.append(raw[:, b["val"]:b["val"] + 1])
                    continue
                val = torch.tanh(raw[:, b["val"]:b["val"] + 1]) * self.clip
                outs.append(val)
                start, end = b["modes"]
                outs.append(torch.softmax(raw[:, start:end], dim=1))
            else:
                outs.append(torch.softmax(raw[:, b["start"]:b["end"]], dim=1))
        return torch.cat(outs, dim=1)

    def forward_with_logits(self, z, cond=None):
        """Same as ``forward`` but also returning the condition-block logits,
        for the generator's direct consistency term."""
        if cond is not None:
            inp = torch.cat([z, cond], dim=1)
        else:
            inp = z
        raw = self.net(inp)
        outs = []
        cond_logits = None
        for b in self.blocks:
            if b["type"] == "num":
                if b.get("kind") == "uniform":
                    outs.append(raw[:, b["val"]:b["val"] + 1])
                    continue
                val = torch.tanh(raw[:, b["val"]:b["val"] + 1]) * self.clip
                outs.append(val)
                start, end = b["modes"]
                outs.append(torch.softmax(raw[:, start:end], dim=1))
            else:
                block = torch.softmax(raw[:, b["start"]:b["end"]], dim=1)
                if getattr(self, "cond_block", None) is not None and b["col"] == self.cond_block["col"]:
                    cond_logits = raw[:, b["start"]:b["end"]]
                outs.append(block)
        return torch.cat(outs, dim=1), cond_logits


class _Discriminator(nn.Module):
    """Discriminator with a critic and an auxiliary condition head (AC-GAN)."""

    def __init__(self, input_dim, hidden, n_layers=2, dropout=0.0, n_cond=0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
        if dropout:
            layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.critic = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, n_cond) if n_cond > 0 else None

    def forward(self, x):
        h = self.net(x)
        out_critic = self.critic(h)
        out_aux = self.aux(h) if self.aux is not None else None
        return out_critic, out_aux


def _clone_discriminator(src: nn.Module, hidden: int, n_layers: int, dropout: float,
                         input_dim: int, n_cond: int) -> _Discriminator:
    """Clone WITHOUT Opacus hooks so the generator step does not touch the DP accounting.

    ``src`` is the GradSampleModule/SampleModule returned by Opacus, which can be
    nested; it is unwrapped down to the ``_Discriminator`` with the same weight
    tensors. The generator gradients flow through the clone (same function), and
    the DP discriminator never accumulates gradients outside its own step.
    """
    inner = src
    while hasattr(inner, "_module") or hasattr(inner, "module"):
        inner = inner._module if hasattr(inner, "_module") else inner.module
    clone = _Discriminator(input_dim, hidden, n_layers, dropout, n_cond)
    clone.load_state_dict(inner.state_dict())
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


@register_generator("dp-gan",
                    description="Conditional tabular GAN with DP-SGD (Opacus) and accounted epsilon",
                    supports=("tabular",))
class DPSGDGenerator(BaseSynthesizer):
    """Tabular generator with a formal differential privacy guarantee.

    Parameters
    ----------
    privacy:
        ``DPSGD`` mechanism with target ``epsilon``/``delta``. If
        ``noise_multiplier`` is set, that noise is used; otherwise Opacus computes
        it to reach the budget from epochs/batch_size/n_samples.
    num_modes:
        Gaussian Mixture modes per numeric column. ``1`` = plain z-score.
        A low value (3-5) captures multimodality; raising it too much burdens the
        generator with extra dimensions.
    condition_column:
        Categorical column conditioning generation. ``None`` picks the most
        imbalanced one (lowest entropy). The generator step samples the condition
        balanced (CTGAN-style) so minority classes are learned, and generation
        samples with the real empirical frequency.
    aux_lambda:
        Weight of the auxiliary losses (discriminator classifier and direct
        consistency between the generated class and its condition).
    generator_steps:
        Generator steps per discriminator step (the DP budget only counts the
        discriminator).
    ecdf_epsilon:
        DP budget for the marginals ECDFs (only with ``numeric="uniform"``).
        Distributed equally across numeric columns (Laplace histogram,
        parallel composition by bins and sequential across columns). The
        synthesizer's total guarantee is the sequential composition of this
        budget with the training one: ``total_epsilon = epsilon(accumulated) +
        ecdf_epsilon`` (the report exposes it in ``accountant.ecdf_epsilon``/
        ``total_epsilon``). ``None`` uses the raw empirical ECDF (no formal
        guarantee on the marginal).
    """

    name = "dp-gan"
    dp_capable = True

    def __init__(
        self,
        privacy: DPSGD | None = None,
        epochs: int = 100,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        layers: int = 2,
        learning_rate: float = 2e-4,
        batch_size: int = 128,
        dropout: float = 0.0,
        num_modes: int = 3,
        clip_value: float = 3.0,
        condition_column: str | None = None,
        aux_lambda: float = 1.0,
        generator_steps: int = 2,
        numeric: str = "mode",
        rectify_marginals: bool = False,
        ecdf_epsilon: float | None = None,
        ecdf_bins: int = 200,
        ecdf_bounds: tuple[float, float] | None = None,
        label_smoothing: float = 0.0,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.privacy = privacy if privacy is not None else DPSGD()
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_modes = num_modes
        self.clip_value = clip_value
        self.condition_column = condition_column
        self.aux_lambda = aux_lambda
        self.generator_steps = max(1, int(generator_steps))
        self.numeric = numeric
        self.rectify_marginals = bool(rectify_marginals)
        self.label_smoothing = float(label_smoothing)
        self._rng = np.random.default_rng(random_state)
        self._seed = int(random_state)
        torch.manual_seed(random_state)
        self.accounted_epsilon: float | None = None
        self._disc_steps_accounted: int | None = None
        self._disc_steps_actual: int = 0
        self._generator: _Generator | None = None
        self._encoder = ModeEncoder(num_modes=num_modes, clip_value=clip_value,
                                    condition_column=condition_column,
                                    numeric=self.numeric,
                                    dp_ecdf_epsilon=ecdf_epsilon,
                                    ecdf_bins=ecdf_bins,
                                    ecdf_bounds=ecdf_bounds)
        self.ecdf_epsilon = self._encoder.dp_ecdf_epsilon

    # ------------------------------------------------------------------
    # DP training
    # ------------------------------------------------------------------
    def _expected_noise(self) -> str:
        return (
            f"target epsilon {self.privacy.epsilon}, delta {self.privacy.delta}, "
            f"n={self._n}, epochs={self.epochs}, batch={self.batch_size}"
        )

    def fit(self, data: pd.DataFrame, **kwargs) -> "DPSGDGenerator":
        self._encoder.fit(data)
        X = torch.from_numpy(self._encoder.transform(data))
        self._n = X.shape[0]
        logger.info("dp-gan encodings: %d dims, condition '%s' (%d classes), %d modes/num",
                    self._encoder.total_dims, self._encoder.condition_column_,
                    self._encoder.n_cond, self._encoder.num_modes)

        conds = self._encoder.condition_vectors(data)
        if conds is not None:
            target = torch.from_numpy(conds)
        else:
            target = torch.zeros(self._n, 1)
        loader = DataLoader(TensorDataset(X, target), batch_size=self.batch_size, shuffle=True)

        generator = _Generator(self.latent_dim, self.hidden_dim, self._encoder, self.layers)
        discriminator = _Discriminator(self._encoder.total_dims, self.hidden_dim,
                                       self.layers, self.dropout, self._encoder.n_cond)
        gen_opt = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        engine = PrivacyEngine()
        bud = self.privacy
        kwargs_dp = {
            "module": discriminator,
            "optimizer": disc_opt,
            "data_loader": loader,
            "max_grad_norm": bud.max_grad_norm,
            "batch_first": True,
            "poisson_sampling": True,
            "clipping": "flat",
        }
        if bud.noise_multiplier is None:
            disc, disc_opt, loader = engine.make_private_with_epsilon(
                target_epsilon=bud.epsilon, target_delta=bud.delta,
                epochs=self.epochs, **kwargs_dp,
            )
            bud.used_noise_multiplier = float(disc_opt.noise_multiplier)
        else:
            disc, disc_opt, loader = engine.make_private(
                noise_multiplier=bud.noise_multiplier, **kwargs_dp,
            )

        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss() if self._encoder.n_cond > 0 else None
        logger.info("DP-GAN adjusting epsilon %s -> noise %.3f (clipping %.2f)",
                    self._expected_noise(), bud.used_noise_multiplier, bud.max_grad_norm)

        self._disc_steps_actual = 0
        for epoch in range(self.epochs):
            probe = _clone_discriminator(disc, self.hidden_dim, self.layers, self.dropout,
                                         self._encoder.total_dims, self._encoder.n_cond)
            for batch in loader:
                params = batch if isinstance(batch, (list, tuple)) else [batch]
                x_real, targets = params[0].to(DEVICE), params[1].to(DEVICE)
                if self._encoder.n_cond > 0:
                    cond_r = targets
                    cond_idx = cond_r.argmax(dim=1)
                else:
                    cond_r, cond_idx = None, None
                b = x_real.shape[0]

                z = torch.randn(b, self.latent_dim, device=DEVICE)
                fake, _ = generator.forward_with_logits(z, cond_r)

                disc_opt.zero_grad()
                crit_real, aux_real = disc(x_real)
                crit_fake, aux_fake = disc(fake.detach())
                ls = self.label_smoothing
                loss_d = bce(crit_real, (1.0 - ls) * torch.ones_like(crit_real)) + \
                    bce(crit_fake, ls * torch.ones_like(crit_fake))
                if ce is not None:
                    loss_d = loss_d + self.aux_lambda * (ce(aux_real, cond_idx) + ce(aux_fake, cond_idx))
                loss_d.backward()
                disc_opt.step()
                self._disc_steps_actual += 1

                z2 = torch.randn(b, self.latent_dim, device=DEVICE)
                if ce is not None:
                    # balanced condition in the generator step (CTGAN-style):
                    # by replicating the real batch (90/10) the generator never
                    # learns minority classes. The discriminator does use the real
                    # condition so the fake marginal frequency does not explode.
                    cond_g = torch.eye(self._encoder.n_cond, device=DEVICE)[
                        torch.randint(self._encoder.n_cond, (b,), device=DEVICE)]
                    cond_idx_g = cond_g.argmax(dim=1)
                else:
                    cond_g, cond_idx_g = None, None
                for _ in range(self.generator_steps):
                    gen_opt.zero_grad()
                    if cond_g is not None:
                        fake_g, cond_logits = generator.forward_with_logits(z2, cond_g)
                    else:
                        fake_g, cond_logits = generator(z2, None), None
                    crit_g, aux_g = probe(fake_g)
                    loss_g = bce(crit_g, torch.ones_like(crit_g))
                    loss_cond = torch.tensor(0.0, device=DEVICE)
                    if ce is not None:
                        loss_g = loss_g + self.aux_lambda * ce(aux_g, cond_idx_g)
                        if cond_logits is not None:
                            loss_cond = ce(cond_logits, cond_idx_g)
                            loss_g = loss_g + self.aux_lambda * loss_cond
                    loss_g.backward()
                    gen_opt.step()
                if ce is not None:
                    loss_cond_epoch = loss_cond.item()

            if epoch % max(1, self.epochs // 5) == 0:
                logger.debug("epoch %d/%d  loss_d=%.3f loss_g=%.3f cond_ce=%.3f",
                             epoch + 1, self.epochs, loss_d.item(), loss_g.item(),
                             loss_cond_epoch if ce is not None else float("nan"))

        eps = engine.get_epsilon(bud.delta)
        eps = eps[0] if isinstance(eps, tuple) else eps
        self.accounted_epsilon = float(eps)
        self._disc_steps_accounted = int(sum(e[2] for e in engine.accountant.history))
        self._generator = generator
        self._fitted = True
        logger.info("Real accumulated epsilon (RDP accountant): %.3f (target %.3f) "
                    "in %d DP steps", self.accounted_epsilon, bud.epsilon,
                    self._disc_steps_accounted)
        return self

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        if not self._fitted or self._generator is None:
            raise RuntimeError("DPSGDGenerator is not fitted: call fit(real_data) first.")
        z = torch.randn(num_rows, self.latent_dim, device=DEVICE)
        cond = self._encoder.sample_conditions(num_rows, self._rng)
        cond_t = torch.from_numpy(cond).to(DEVICE) if cond is not None else None
        with torch.no_grad():
            out = self._generator(z, cond_t).cpu().numpy()
        if self.rectify_marginals and self._encoder is not None:
            out = self._encoder.rectify(out)
        return self._encoder.inverse(out)

    # ------------------------------------------------------------------
    # DP assurance
    # ------------------------------------------------------------------
    def assert_dp(self, declared_epsilon: float | None = None, *,
                  tolerance: float = 0.05, delta: float | None = None) -> DpAssurance:
        """Validate that the declared DP guarantee is not exceeded (steps + budget)."""
        return assert_dp(self, declared_epsilon, tolerance=tolerance, delta=delta)

    # ------------------------------------------------------------------
    # persistence: one file with config + encoder + weights + DP accounting
    # ------------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "num_modes": self.num_modes,
            "clip_value": self.clip_value,
            "condition_column": self.condition_column,
            "aux_lambda": self.aux_lambda,
            "generator_steps": self.generator_steps,
            "numeric": self.numeric,
            "rectify_marginals": self.rectify_marginals,
            "ecdf_epsilon": self.ecdf_epsilon,
            "ecdf_bins": self._encoder.ecdf_bins,
            "ecdf_bounds": self._encoder.ecdf_bounds,
            "label_smoothing": self.label_smoothing,
            "random_state": int(self._seed),
        }

    def save(self, path: str | Path) -> Path:
        """Persist config + encoder + weights + DP accounting in one file.

        The real accumulated epsilon and the applied noise are kept: when
        reloading, the declared DP guarantee is the same as when trained.
        """
        path = Path(path)
        payload = {
            "version": 2,
            "class": self.__class__.__name__,
            "name": self.name,
            "fitted": self._fitted,
            "params": self.get_params(),
            "privacy": self.privacy,
            "accounted_epsilon": self.accounted_epsilon,
            "n": getattr(self, "_n", None),
            "disc_steps_accounted": getattr(self, "_disc_steps_accounted", None),
            "disc_steps_actual": getattr(self, "_disc_steps_actual", 0),
            "encoder": self._encoder,
            "generator": self._generator.state_dict() if self._generator is not None else None,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "DPSGDGenerator":
        """Rebuild a trained generator and its DP accounting from ``path``."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        params = dict(payload.get("params", {}))
        params.update(overrides)
        privacy = payload.get("privacy")
        obj = cls(privacy=privacy, **params)
        obj._seed = params.get("random_state", 0)
        obj._rng = np.random.default_rng(obj._seed)
        obj.accounted_epsilon = payload.get("accounted_epsilon")
        obj._n = payload.get("n")
        obj._disc_steps_accounted = payload.get("disc_steps_accounted")
        obj._disc_steps_actual = payload.get("disc_steps_actual", 0)
        obj._encoder = payload.get("encoder")
        obj._fitted = bool(payload.get("fitted"))
        if payload.get("generator") is not None:
            generator = _Generator(obj.latent_dim, obj.hidden_dim, obj._encoder, obj.layers)
            generator.load_state_dict(payload["generator"])
            obj._generator = generator
        return obj