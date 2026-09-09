from .passthrough import PassThroughRuntime


class QueryRobustRuntime(PassThroughRuntime):
    """Logical runtime for QR's cache-owned paged decode view."""
