"""Writing a measurement back into the configuration file.

Calibration measures a threshold and the agent then has to be told about it.
Doing that by hand at 09:40, minutes before an entry, is the kind of step that
goes wrong exactly once and takes a session with it. So this does it, and the
care is all in what it refuses to do.

**It edits lines, it does not rewrite the file.** Loading the YAML and dumping
it back would drop every comment, and the comments here are not decoration:
they carry the MEASURED and BOUND markings that the provenance lists are checked
against, and a test fails when the two disagree. So the value line is found and
rewritten in place and every other byte of the file is left exactly as it was.

**Both halves land or neither does.** A measurement is two edits, the value and
the removal of that key from the blocking list, and a file carrying one without
the other is worse than a file carrying neither. A value updated while still
listed leaves the agent standing down over a number that was in fact measured. A
key removed while its value is still a guess is precisely the failure the
calibration check exists to prevent. Both are done to a string in memory, so a
failure anywhere means nothing reaches disk.

**The write is atomic.** convex/config.py re-reads whenever the mtime moves, so
a cycle running in another process can read this file at any instant, including
halfway through a write. The new text goes to a temporary file alongside and is
moved into place in one step.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date
from pathlib import Path

from convex.errors import ConfigError


def apply_measurement(
    text: str, key: str, value: float, when: date, by: str, places: int = 4
) -> str:
    """Return the file with one key measured, or raise having changed nothing.

    ``key`` is dotted, as everything else in this project addresses config. The
    section and the leaf are located by structure rather than by a bare name
    search, because several sections carry keys of the same name and rewriting
    the wrong one would be silent.
    """
    section, _, leaf = key.rpartition(".")
    if not section or not leaf:
        raise ConfigError(f"{key!r} is not a dotted config key")

    updated = _rewrite_value(text, section, leaf, value, when, by, places)
    return _drop_from_hypotheses(updated, key)


def refresh_measurement(
    text: str, key: str, value: float, when: date, by: str, places: int = 4
) -> str:
    """Update a previously attested measurement without reintroducing a hypothesis.

    A live session remeasures its liquidity threshold every day. The first
    measurement clears the blocking provenance entry; later measurements update
    the dated value but must not fail merely because that entry is already gone.
    """
    section, _, leaf = key.rpartition(".")
    if not section or not leaf:
        raise ConfigError(f"{key!r} is not a dotted config key")
    updated = _rewrite_value(text, section, leaf, value, when, by, places)
    try:
        return _drop_from_hypotheses(updated, key)
    except ConfigError as error:
        if "not listed under provenance.hypothesis" not in str(error):
            raise
        return updated


def _rewrite_value(
    text: str, section: str, leaf: str, value: float, when: date, by: str, places: int
) -> str:
    lines = text.splitlines(keepends=True)
    inside = False
    for index, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if re.match(rf"^{re.escape(section)}\s*:\s*$", stripped):
            inside = True
            continue
        # A new top-level key ends the section. Indented lines and blank lines
        # and comments do not.
        if inside and stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            break
        if not inside:
            continue

        match = re.match(rf"^(\s+){re.escape(leaf)}\s*:\s*([^#\n]*?)\s*(#.*)?$", stripped)
        if match:
            indent = match.group(1)
            rendered = f"{indent}{leaf}: {value:.{places}f}"
            comment = f"# MEASURED {when.isoformat()} by {by}"
            # Keep the comment column the file already uses where it fits, so a
            # measured line still reads in the same column as its neighbours.
            pad = max(36 - len(rendered), 1)
            lines[index] = f"{rendered}{' ' * pad}{comment}\n"
            return "".join(lines)

    raise ConfigError(f"{section}.{leaf} was not found in the configuration file")


def _drop_from_hypotheses(text: str, key: str) -> str:
    lines = text.splitlines(keepends=True)
    inside_provenance = False
    inside_list = False
    for index, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if re.match(r"^provenance\s*:\s*$", stripped):
            inside_provenance = True
            continue
        if inside_provenance and stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            break
        if not inside_provenance:
            continue
        if re.match(r"^\s+hypothesis\s*:\s*$", stripped):
            inside_list = True
            continue
        # Any other key at that depth closes the list.
        if inside_list and re.match(r"^\s{2}\w[\w_]*\s*:\s*$", stripped):
            inside_list = False
        if inside_list and re.match(rf"^\s+-\s+{re.escape(key)}\s*$", stripped):
            del lines[index]
            return "".join(lines)

    raise ConfigError(
        f"{key} is not listed under provenance.hypothesis, so there is nothing "
        "to clear. It may already have been measured."
    )


def write_atomically(path: Path, text: str) -> None:
    """Replace the file in one step, never leaving a half-written config."""
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
