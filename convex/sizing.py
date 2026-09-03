"""Position sizing.

Law 11: size is the output of one function with no override parameter. There is
no confidence multiplier, no conviction weighting and no discretionary scaling,
because the research found that hard mapping beats confidence-weighted sizing.
The classifier decides whether a structure trades. It never decides how much.

Law 8: the inputs are the worst case and the tail, never the expected profit. A
structure with a spectacular mean and an unacceptable one-percent tail sizes
small or not at all, which is the entire point.

Three constraints bind, and the function reports which one did:

  risk budget    a fixed fraction of equity divided by the worst case per lot
  tail budget    the projected portfolio ES(1%) ceiling, less what is in use
  buying power   what the account will actually let through
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from convex.config import Config
from convex.edge import EdgeEstimate
from convex.errors import ConfigError


@dataclass(frozen=True)
class PortfolioState:
    """What the account already has at risk when this size is computed."""

    equity: float
    buying_power: float
    open_structures: int
    es_in_use: float

    def __post_init__(self) -> None:
        if self.equity <= 0.0:
            raise ConfigError(f"equity must be positive to size against, found {self.equity}")
        if self.es_in_use < 0.0:
            raise ConfigError(f"expected shortfall in use cannot be negative: {self.es_in_use}")


@dataclass(frozen=True)
class SizeDecision:
    """The size, and an account of why it is that size and not larger."""

    contracts: int
    risk_budget: float
    max_loss_per_contract: float
    es_per_contract: float
    es_headroom: float
    buying_power_per_contract: float
    binding_constraint: str

    @property
    def trades(self) -> bool:
        return self.contracts > 0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "contracts": self.contracts,
            "risk_budget": round(self.risk_budget, 2),
            "max_loss_per_contract": round(self.max_loss_per_contract, 2),
            "es_per_contract": round(self.es_per_contract, 2),
            "es_headroom": round(self.es_headroom, 2),
            "binding_constraint": self.binding_constraint,
        }


def size_position(
    estimate: EdgeEstimate,
    portfolio: PortfolioState,
    config: Config,
) -> SizeDecision:
    """Contracts to trade. One function, no override, deterministic.

    ``estimate`` must be priced at one contract: everything here scales
    linearly in size, so a single-lot evaluation is the honest unit and
    multiplying it is exact rather than approximate.
    """
    if estimate.contracts != 1:
        raise ConfigError(
            f"sizing expects a single-lot estimate, received {estimate.contracts} contracts"
        )

    risk_pct = config.float_("risk.risk_pct_per_structure")
    es_cap_pct = config.float_("risk.portfolio_es_cap_pct")
    max_contracts = config.int_("risk.max_contracts_per_structure")
    if not 0.0 < risk_pct < 1.0 or not 0.0 < es_cap_pct < 1.0:
        raise ConfigError("risk fractions must lie strictly between 0 and 1")
    if max_contracts <= 0:
        raise ConfigError("risk.max_contracts_per_structure must be positive")

    max_loss = estimate.profile.max_loss
    if max_loss <= 0.0:
        raise ConfigError(f"cannot size against a worst case of {max_loss}")

    risk_budget = portfolio.equity * risk_pct
    by_risk = math.floor(risk_budget / max_loss)

    es_per_contract = estimate.expected_shortfall
    es_headroom = portfolio.equity * es_cap_pct - portfolio.es_in_use
    by_tail = math.floor(es_headroom / es_per_contract) if es_per_contract > 0.0 else by_risk

    # The buying power a defined-risk structure consumes is its worst case; a
    # credit structure also frees the credit, which is deliberately ignored so
    # the constraint errs towards refusing rather than towards trading.
    by_buying_power = math.floor(portfolio.buying_power / max_loss)

    limits = {
        "risk_budget": by_risk,
        "portfolio_tail": by_tail,
        "buying_power": by_buying_power,
        "profile_contract_cap": max_contracts,
    }
    binding = min(limits, key=lambda name: limits[name])
    contracts = max(0, min(limits.values()))

    return SizeDecision(
        contracts=contracts,
        risk_budget=risk_budget,
        max_loss_per_contract=max_loss,
        es_per_contract=es_per_contract,
        es_headroom=es_headroom,
        buying_power_per_contract=max_loss,
        binding_constraint=binding,
    )
