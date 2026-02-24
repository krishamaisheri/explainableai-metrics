"""
Explainability metrics package.
Each submodule exposes a ``compute(...)`` function returning a float ∈ [0, 1].
"""

from metrics.iacs import compute as compute_iacs
from metrics.icr import compute as compute_icr
from metrics.ircs import compute as compute_ircs
from metrics.edas import compute as compute_edas
from metrics.secs import compute as compute_secs
from metrics.pgss import compute as compute_pgss
from metrics.esi import compute as compute_esi
from metrics.edr import compute as compute_edr

METRIC_REGISTRY = {
    "IACS": compute_iacs,
    "ICR":  compute_icr,
    "IRCS": compute_ircs,
    "EDAS": compute_edas,
    "SECS": compute_secs,
    "PGSS": compute_pgss,
    "ESI":  compute_esi,
    "EDR":  compute_edr,
}
