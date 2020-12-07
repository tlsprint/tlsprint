from . import dedup_per_implementation
from . import dedup_per_tls
from . import total_models
from . import tree_sizes

TYPE_HANDLERS = {
    "total-models": total_models.summary,
    "dedup-per-tls": dedup_per_tls.summary,
    "dedup-per-implementation": dedup_per_implementation.summary,
    "tree-sizes": tree_sizes.summary,
}
TYPES = TYPE_HANDLERS.keys()
