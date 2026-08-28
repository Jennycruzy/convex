"""Failure types.

Law 3: fail loudly, never silently. Every module in this package raises one of
these instead of returning a default, a None, or a zero. There are no empty
except blocks anywhere in this codebase and no `or 0` fallbacks.
"""


class ConvexError(Exception):
    """Base for every error CONVEX raises."""


class ConfigError(ConvexError):
    """Configuration is missing, malformed, or internally inconsistent."""


class CredentialsError(ConvexError):
    """Alpaca or Featherless credentials are absent or rejected."""


class DataError(ConvexError):
    """Market data is missing, stale, or structurally unusable.

    Raised for an absent Greek, a crossed or empty quote, a chain that does not
    contain the requested expiry. Never downgrade one of these into a default.
    """


class StaleDataError(DataError):
    """A quote or chain snapshot is older than the staleness budget."""


class UndefinedRiskError(ConvexError):
    """A structure's maximum loss is not computable or not bounded.

    Law 5. The agent is structurally incapable of submitting an order whose max
    loss it cannot compute, because pricing a structure goes through the max
    loss calculator and this is what that calculator raises.
    """


class ExecutionError(ConvexError):
    """An order was rejected, partially filled, or otherwise did not complete."""


class KillSwitchEngaged(ConvexError):
    """The kill switch file is present. No new risk may be taken."""
