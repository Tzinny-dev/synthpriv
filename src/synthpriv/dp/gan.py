"""GAN tabular entrenado con DP-SGD (Opacus).

Construccion clasica de DP-GAN: solo el discriminador ve los datos y se entrena
con DP-SGD (recorte + ruido en los gradientes). El generador es post-proceso
del discriminador, asi que el resultado es DP con el epsilon contabilizado.

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
from synthpriv.dp.encoder import TabularEncoder
from synthpriv.privacy.mechanisms import DPSGD
from synthpriv.utils import get_logger

# Avisos benignos y abundantes de Opacus/Torch durante el entrenamiento DP.
_BN_ALERTAS = ("Full backward hook", "Secure RNG turned off", "Optimal order is the largest alpha")
for _m in _BN_ALERTAS:
    warnings.filterwarnings("ignore", message=f"{_m}.*")

logger = get_logger("dp")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gen_head(latent, hidden, out_dim, n_layers):
    layers = [nn.Linear(latent, hidden), nn.ReLU()]
    for _ in range(max(0, n_layers - 2)):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


def _disc_head(input_dim, hidden, n_layers, dropout):
    layers = [nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2)]
    for _ in range(max(0, n_layers - 2)):
        layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
    if dropout:
        layers += [nn.Dropout(dropout)]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


class _Generator(nn.Module):
    """MLP de ``latent`` a la salida del encoder.

    Devuelve el tensor ya "montado": numericas en bruto + softmax por bloque
    categorico, espacio identico al codificado (``encoder.total_dims``).
    """

    def __init__(self, latent, hidden, encoder: TabularEncoder, n_layers=2):
        super().__init__()
        self.encoder = encoder
        self.net = _gen_head(latent, hidden, encoder.total_dims, n_layers)

    def forward(self, z):
        raw = self.net(z)
        num = raw[:, : self.encoder.num_dims]
        outs = [num]
        for start, end in self.encoder.categorical_spans:
            outs.append(torch.softmax(raw[:, start:end], dim=1))
        return torch.cat(outs, dim=1)


class _Discriminator(nn.Module):
    def __init__(self, input_dim, hidden, n_layers=2, dropout=0.0):
        super().__init__()
        self.net = _disc_head(input_dim, hidden, n_layers, dropout)

    def forward(self, x):
        return self.net(x)


def _clone_discriminator(src: nn.Module, hidden: int, n_layers: int, dropout: float,
                         input_dim: int) -> _Discriminator:
    """Copia SIN hooks de Opacus para que el paso del generador no toque la contabilidad DP.

    ``src`` es el GradSampleModule/SampleModule devuelto por Opacus, que puede
    estar anidado; se desenvuelve hasta el ``_Discriminator`` con los mismos
    tensores de pesos. Los gradientes del generador fluyen por el clon (misma
    funcion), y el discriminador DP nunca acumula gradientes fuera de su paso.
    """
    inner = src
    while hasattr(inner, "_module") or hasattr(inner, "module"):
        inner = inner._module if hasattr(inner, "_module") else inner.module
    clone = _Discriminator(input_dim, hidden, n_layers, dropout)
    clone.load_state_dict(inner.state_dict())
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


@register_generator("dp-gan",
                    description="GAN MLP tabular con DP-SGD (Opacus) y epsilon contabilizado",
                    supports=("tabular",))
class DPSGDGenerator(BaseSynthesizer):
    """Generador tabular con garantia formal de privacidad diferencial.

    Parameters
    ----------
    privacy:
        Mecanismo ``DPSGD`` con ``epsilon``/``delta`` objetivo. Si se fija
        ``noise_multiplier`` se usa ese ruido; si no, Opacus lo calcula para
        alcanzar el presupuesto a partir de epochs/batch_size/n_muestras.
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
        self._rng = np.random.default_rng(random_state)
        torch.manual_seed(random_state)
        self.accounted_epsilon: float | None = None
        self._generator: _Generator | None = None
        self._encoder = TabularEncoder()

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

        target = torch.zeros(self._n, 1)
        loader = DataLoader(TensorDataset(X, target), batch_size=self.batch_size, shuffle=True)

        generator = _Generator(self.latent_dim, self.hidden_dim, self._encoder, self.layers)
        discriminator = _Discriminator(self._encoder.total_dims, self.hidden_dim,
                                       self.layers, self.dropout)
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
        logger.info("DP-GAN ajustando epsilon %s -> ruido %.3f (clipping %.2f)",
                    self._expected_noise(), bud.used_noise_multiplier, bud.max_grad_norm)

        for epoch in range(self.epochs):
            probe = _clone_discriminator(disc, self.hidden_dim, self.layers, self.dropout,
                                         self._encoder.total_dims)
            for batch in loader:
                params = batch if isinstance(batch, (list, tuple)) else [batch]
                x_real, targets = params[0].to(DEVICE), params[1].to(DEVICE)
                b = x_real.shape[0]

                z = torch.randn(b, self.latent_dim, device=DEVICE)
                fake = generator(z)

                disc_opt.zero_grad()
                loss_d_real = bce(disc(x_real), torch.ones_like(targets))
                loss_d_fake = bce(disc(fake.detach()), torch.zeros_like(targets))
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                disc_opt.step()

                z2 = torch.randn(b, self.latent_dim, device=DEVICE)
                gen_opt.zero_grad()
                loss_g = bce(probe(generator(z2)), torch.ones_like(targets))
                loss_g.backward()
                gen_opt.step()

            if epoch % max(1, self.epochs // 5) == 0:
                logger.debug("epoch %d/%d  loss_d=%.3f loss_g=%.3f",
                             epoch + 1, self.epochs, loss_d.item(), loss_g.item())

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
        with torch.no_grad():
            out = self._generator(z).cpu().numpy()
        return self._encoder.inverse(out)