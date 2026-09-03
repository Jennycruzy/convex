"""The structure families CONVEX is allowed to trade.

Five families, and the fifth is standing down. Symmetric butterflies and iron
condors are absent by design: they are the structures the research measures at
a positive gross Sharpe and a negative net Sharpe, and refusing to build them
is the thesis rather than an omission.
"""

from convex.structures.base import Candidate, Family, chain_index
from convex.structures.builders import BUILDABLE_FAMILIES, build_candidates

__all__ = ["BUILDABLE_FAMILIES", "Candidate", "Family", "build_candidates", "chain_index"]
