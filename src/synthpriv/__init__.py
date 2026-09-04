"""synthpriv: generacion de datos sinteticos preservando privacidad."""

from synthpriv.core.registry import (
    GeneratorNotFoundError,
    build_generator,
    get_generator,
    list_generators,
    register_generator,
)
from synthpriv.generators.tabular import (
    CTGANGenerator,
    CopulaGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)
from synthpriv.dp.gan import DPSGDGenerator
from synthpriv.pipeline import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD, NoPrivacy, PrivacyMechanism

__version__ = "0.1.0"

__all__ = [
    "GeneratorNotFoundError",
    "PrivacyPreservingSynthesizer",
    "CTGANGenerator",
    "TVAEGenerator",
    "CopulaGANGenerator",
    "GaussianCopulaGenerator",
    "DPSGDGenerator",
    "PrivacyMechanism",
    "NoPrivacy",
    "DPSGD",
    "register_generator",
    "get_generator",
    "build_generator",
    "list_generators",
    "__version__",
]