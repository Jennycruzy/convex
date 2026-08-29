"""Hot-loadable configuration.

Every threshold in the system lives in config/convex.yaml. Nothing reads a
constant from source. load() re-reads the file whenever its mtime has moved, so
editing a threshold between decision cycles takes effect without a restart.

Law 3 applies to lookups: a missing key raises ConfigError rather than
returning a default, because a silently defaulted risk threshold is a risk
threshold nobody chose.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from convex.errors import ConfigError

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("CONVEX_CONFIG", Path(__file__).resolve().parent.parent / "config" / "convex.yaml")
)

_lock = threading.Lock()
_cache: dict[Path, tuple[float, "Config"]] = {}


@dataclass(frozen=True)
class Config:
    """An immutable snapshot of config/convex.yaml."""

    path: Path
    loaded_mtime: float
    values: dict[str, Any]

    def get(self, dotted: str) -> Any:
        """Return the value at a dotted path, raising if any segment is absent."""
        node: Any = self.values
        walked: list[str] = []
        for segment in dotted.split("."):
            walked.append(segment)
            if not isinstance(node, dict) or segment not in node:
                raise ConfigError(
                    f"config key {dotted!r} is missing at {'.'.join(walked)!r} in {self.path}"
                )
            node = node[segment]
        return node

    def float_(self, dotted: str) -> float:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config key {dotted!r} must be a number, found {value!r}")
        return float(value)

    def int_(self, dotted: str) -> int:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config key {dotted!r} must be an integer, found {value!r}")
        return value

    def str_(self, dotted: str) -> str:
        value = self.get(dotted)
        if not isinstance(value, str):
            raise ConfigError(f"config key {dotted!r} must be a string, found {value!r}")
        return value

    def list_(self, dotted: str) -> list[Any]:
        value = self.get(dotted)
        if not isinstance(value, list):
            raise ConfigError(f"config key {dotted!r} must be a list, found {value!r}")
        return value

    def path_(self, dotted: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        return (self.path.parent.parent / self.str_(dotted)).resolve()

    def _provenance(self, kind: str) -> tuple[str, ...]:
        listed = self.get(f"provenance.{kind}")
        if not isinstance(listed, list) or not all(isinstance(key, str) for key in listed):
            raise ConfigError(
                f"provenance.{kind} must be a list of config keys in {self.path}"
            )
        for key in listed:
            self.get(key)
        return tuple(listed)

    def hypotheses(self) -> tuple[str, ...]:
        """Unmeasured keys that could be wrong in the dangerous direction.

        A key named here that does not exist is a typo, and a typo in this list
        would quietly shrink the set of values being guarded, so it raises.
        """
        return self._provenance("hypothesis")

    def bounds(self) -> tuple[str, ...]:
        """Unmeasured keys deliberately set beyond what they can plausibly be.

        Trading on one of these is safe in a way that trading on a hypothesis is
        not. Over-stating what execution costs refuses candidates that were
        marginally worth taking; it never admits one that was not. Some of these
        cannot be measured until a fill exists, so treating them the same as a
        guess would leave the agent unable to trade and therefore unable to ever
        measure them.
        """
        return self._provenance("conservative_bound")

    def unmeasured(self, *keys: str) -> tuple[str, ...]:
        """Which of the given keys are still hypotheses, in the order asked."""
        listed = set(self.hypotheses())
        return tuple(key for key in keys if key in listed)


def load(path: Path | str | None = None) -> Config:
    """Return the current configuration, re-reading the file if it has changed."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ConfigError(f"configuration file not found: {resolved}")

    mtime = resolved.stat().st_mtime
    with _lock:
        cached = _cache.get(resolved)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    try:
        parsed = yaml.safe_load(resolved.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved} is not valid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{resolved} must contain a mapping at the top level")

    config = Config(path=resolved, loaded_mtime=mtime, values=parsed)
    with _lock:
        _cache[resolved] = (mtime, config)
    return config
