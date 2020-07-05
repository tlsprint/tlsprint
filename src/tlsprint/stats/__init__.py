from . import dedup_per_implementation
from . import dedup_per_tls
from . import total_models

TYPE_HANDLERS = {
    "total-models": total_models.summary,
    "dedup-per-tls": dedup_per_tls.summary,
    "dedup-per-implementation": dedup_per_implementation.summary,
}
TYPES = TYPE_HANDLERS.keys()
