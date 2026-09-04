"""GAN tabular entrenado con DP-SGD (Opacus), condicionado por clase.

Construccion clasica de DP-GAN: solo el discriminador ve los datos y se entrena
con DP-SGD (recorte + ruido en los gradientes). El generador es post-proceso
del discriminador, asi que el resultado es DP con el epsilon contabilizado.

Para utilidad se usa el esquema AC-GAN: el generador recibe un vector de
condicion (clase de la columna mas imbalanced) y el discriminador tiene una
cabeza auxiliar que debe predecirla. La condicion se muestrea balanceada en el
paso del generador (estilo CTGAN) para que las clases minoritarias no queden
colapsadas a la mayoritaria; al muestrear se usa la frecuencia empirica para
respetar el marginal. Las numericas usan normalizacion mode-specific
(``ModeEncoder``) para no aplastar modos.

El accountant (RDP) traduce ruido, epochs y tamano de muestra al epsilon
acumulado real, que se expone en ``accounted_epsilon`` tras ``fit``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from synthpriv.core.base import BaseSynthesizer
from synthpriv.core.registry import register_generator
from synthpriv.dp.encoder import ModeEncoder
from synthpriv.privacy.mechanisms import DPSGD
from synthpriv.utils import get_logger

# Avisos benignos y abundantes de Opacus/Torch durante el entrenamiento DP.
_BN_ALERTAS = ("Full backward hook", "Secure RNG turned off", "Optimal order is the largest alpha")
for _m in _BN_ALERTAS:
    warnings.filterwarnings("ignore", message=f"{_m}.*")

logger = get_logger("dp")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Generator(nn.Module):
    """MLP condicional: ``(z, cond)`` -> vector en el espacio del ``ModeEncoder``.

    La salida se monta por bloques: cada numerica aporta ``tanh(valor)`` + softmax
    sobre sus modos, y cada categorica su softmax. Resultado isomorfo al codificado.
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
                val = torch.tanh(raw[:, b["val"]:b["val"] + 1]) * self.clip
                start, end = b["modes"]
                outs += [val, torch.softmax(raw[:, start:end], dim=1)]
            else:
                outs.append(torch.softmax(raw[:, b["start"]:b["end"]], dim=1))
        return torch.cat(outs, dim=1)

    def forward_with_logits(self, z, cond=None):
        """Igual que ``forward`` pero devolviendo ademas los logits del bloque
        de condicion, para el termino de consistencia directa del generador."""
        if cond is not None:
            inp = torch.cat([z, cond], dim=1)
        else:
            inp = z
        raw = self.net(inp)
        outs = []
        cond_logits = None
        for b in self.blocks:
            if b["type"] == "num":
                val = torch.tanh(raw[:, b["val"]:b["val"] + 1]) * self.clip
                start, end = b["modes"]
                outs += [val, torch.softmax(raw[:, start:end], dim=1)]
            else:
                block = torch.softmax(raw[:, b["start"]:b["end"]], dim=1)
                if getattr(self, "cond_block", None) is not None and b["col"] == self.cond_block["col"]:
                    cond_logits = raw[:, b["start"]:b["end"]]
                outs.append(block)
        return torch.cat(outs, dim=1), cond_logits


class _Discriminator(nn.Module):
    """Discriminador con critica y cabeza auxiliar de condicion (AC-GAN)."""

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
    """Copia SIN hooks de Opacus para que el paso del generador no toque la contabilidad DP.

    ``src`` es el GradSampleModule/SampleModule devuelto por Opacus, que puede
    estar anidado; se desenvuelve hasta el ``_Discriminator`` con los mismos
    tensores de pesos. Los gradientes del generador fluyen por el clon (misma
    funcion), y el discriminador DP nunca acumula gradientes fuera de su paso.
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
                    description="GAN condicional tabular con DP-SGD (Opacus) y epsilon contabilizado",
                    supports=("tabular",))
class DPSGDGenerator(BaseSynthesizer):
    """Generador tabular con garantia formal de privacidad diferencial.

    Parameters
    ----------
    privacy:
        Mecanismo ``DPSGD`` con ``epsilon``/``delta`` objetivo. Si se fija
        ``noise_multiplier`` se usa ese ruido; si no, Opacus lo calcula para
        alcanzar el presupuesto a partir de epochs/batch_size/n_muestras.
    num_modes:
        Modos Gaussian Mixture por columna numerica. ``1`` = z-score simple.
        Un valor bajo (3-5) captura multimodalidad; subirlo mucho lastra al
        generador con dimensiones extra.
    condition_column:
        Columna categorica que condiciona la generacion. ``None`` elige la mas
        imbalanced (menor entropia). El paso del generador muestrea la condicion
        de forma balanceada (estilo CTGAN) para que las clases minoritarias se
        aprendan, y al generar se muestrea con la frecuencia empirica real.
    aux_lambda:
        Peso de las perdidas auxiliares (clasificador del discriminador y
        consistencia directa entre la clase generada y su condicion).
    generator_steps:
        Pasos del generador por paso del discriminador (el presupuesto DP solo
        cuenta el discriminador).
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
        self._rng = np.random.default_rng(random_state)
        torch.manual_seed(random_state)
        self.accounted_epsilon: float | None = None
        self._generator: _Generator | None = None
        self._encoder = ModeEncoder(num_modes=num_modes, clip_value=clip_value,
                                    condition_column=condition_column)

    # ------------------------------------------------------------------
    # entrenamiento DP
    # ------------------------------------------------------------------
    def _expected_noise(self) -> str:
        return (
            f"epsilon objetivo {self.privacy.epsilon}, delta {self.privacy.delta}, "
            f"n={self._n}, epochs={self.epochs}, batch={self.batch_size}"
        )

    def fit(self, data: pd.DataFrame, **kwargs) -> "DPSGDGenerator":
        self._encoder.fit(data)
        X = torch.from_numpy(self._encoder.transform(data))
        self._n = X.shape[0]
        logger.info("dp-gan encodings: %d dims, condicion '%s' (%d clases), %d modos/num",
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
        logger.info("DP-GAN ajustando epsilon %s -> ruido %.3f (clipping %.2f)",
                    self._expected_noise(), bud.used_noise_multiplier, bud.max_grad_norm)

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
                loss_d = bce(crit_real, torch.ones_like(crit_real)) + \
                    bce(crit_fake, torch.zeros_like(crit_fake))
                if ce is not None:
                    loss_d = loss_d + self.aux_lambda * (ce(aux_real, cond_idx) + ce(aux_fake, cond_idx))
                loss_d.backward()
                disc_opt.step()

                z2 = torch.randn(b, self.latent_dim, device=DEVICE)
                if ce is not None:
                    # condicion balanceada en el paso del generador (estilo CTGAN):
                    # si replicamos el batch real (90/10) el generador nunca aprende
                    # las clases minoritarias. El discriminador si usa la cond real
                    # para que no explote la frecuencia marginal de los fakes.
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
                             epoch + 1, self.epochs, loss_d.item(), loss_g.item(), loss_cond_epoch)

        eps = engine.get_epsilon(bud.delta)
        eps = eps[0] if isinstance(eps, tuple) else eps
        self.accounted_epsilon = float(eps)
        self._generator = generator
        self._fitted = True
        logger.info("Epsilon acumulado real (RDP accountant): %.3f (objetivo %.3f)",
                    self.accounted_epsilon, bud.epsilon)
        return self

    # ------------------------------------------------------------------
    # muestreo
    # ------------------------------------------------------------------
    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        if not self._fitted or self._generator is None:
            raise RuntimeError("DPSGDGenerator no esta entrenado: llama a fit(real_data) primero.")
        z = torch.randn(num_rows, self.latent_dim, device=DEVICE)
        cond = self._encoder.sample_conditions(num_rows, self._rng)
        cond_t = torch.from_numpy(cond).to(DEVICE) if cond is not None else None
        with torch.no_grad():
            out = self._generator(z, cond_t).cpu().numpy()
        return self._encoder.inverse(out)