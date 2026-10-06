"""Enregistre les familles de détecteurs ; API publique dans regime_lib."""

# Import each family so its concrete methods register with METHOD_REGISTRY.
from regime_lib.methods import entropy  # noqa: F401
from regime_lib.methods import misc  # noqa: F401
from regime_lib.methods import price  # noqa: F401
from regime_lib.methods import statistical  # noqa: F401
from regime_lib.methods import trend  # noqa: F401
from regime_lib.methods import vector  # noqa: F401
from regime_lib.methods import volatility  # noqa: F401
from regime_lib.methods import volume  # noqa: F401
