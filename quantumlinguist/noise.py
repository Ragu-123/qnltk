import pennylane as qml
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NoiseModel:
    """Noise model abstraction for quantum circuits."""

    def __init__(self):
        self._channels = []

    def add_depolarizing(self, p: float, gate: str = "all") -> "NoiseModel":
        """Add a depolarizing channel with probability p, optionally tied to a gate."""
        self._channels.append(("depolarizing", p, gate))
        return self

    def add_amplitude_damping(self, gamma: float) -> "NoiseModel":
        """Add an amplitude damping channel with decay rate gamma."""
        self._channels.append(("amplitude_damping", gamma, "all"))
        return self

    def add_bit_flip(self, p: float) -> "NoiseModel":
        """Add a bit-flip channel with flip probability p."""
        self._channels.append(("bit_flip", p, "all"))
        return self

    def apply_to_backend(self, backend_name: str) -> str:
        """Returns 'default.mixed' if noise channels are set, else original backend."""
        if self._channels:
            return "default.mixed"
        return backend_name

    def get_channels(self) -> list:
        """Return a copy of the registered noise channels."""
        return list(self._channels)
