"""Fast, read-only release gate for the CONVEX hackathon submission.

Run from the repository root:

    .venv/bin/python -m scripts.submission_check --public

The checks are intentionally bounded to complete in about a minute. They do not
read credentials, submit orders, write the ledger, or alter configuration.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "docs/presentation/CONVEX_Hackathon_Deck.pptx"
CONFIG = ROOT / "config/convex.yaml"
FOCUSED_TESTS = (
    "tests/test_dashboard.py",
    "tests/test_gap_continuation.py",
    "tests/test_agent.py",
    "tests/test_directional_backtest.py",
    "tests/test_profile_backtest.py",
)


class CheckFailure(Exception):
    """A release gate failed with a terminal-safe message."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def run(command: list[str], label: str, timeout: int = 45) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
        raise CheckFailure(f"{label} failed:\n" + "\n".join(tail))


def config_check() -> str:
    try:
        raw = yaml.safe_load(CONFIG.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise CheckFailure(f"config cannot be parsed: {error}") from error
    check(
        raw["strategy"]["active_profile"] == "gap_continuation_vertical",
        "unexpected active profile",
    )
    ladder = raw["execution"]["reprice_ticks"]
    check(ladder == [1, 2], f"unexpected reprice ladder: {ladder!r}")
    check(raw["structures"]["enabled"] == [], "legacy generic structures must remain disabled")
    return "policy: gap/VWAP; generic families disabled; retry ladder [1, 2]"


def deck_check() -> str:
    check(DECK.is_file(), f"deck missing: {DECK.relative_to(ROOT)}")
    try:
        with zipfile.ZipFile(DECK) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as error:
        raise CheckFailure(f"deck is not a valid .pptx archive: {error}") from error
    slides = [
        name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")
    ]
    check(len(slides) == 10, f"deck must contain 10 slides, found {len(slides)}")
    check("ppt/presentation.xml" in names, "deck has no presentation manifest")
    return "deck: valid PowerPoint archive with 10 slides"


def http_check(url: str, expected_type: str | None = None) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "convex-submission-check/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            disposition = response.headers.get("Content-Disposition", "")
    except (urllib.error.URLError, TimeoutError) as error:
        raise CheckFailure(f"{url} is unavailable: {error}") from error
    check(status == 200, f"{url} returned HTTP {status}")
    if expected_type is not None:
        check(expected_type in content_type, f"{url} returned {content_type!r}")
        check("attachment" in disposition, f"{url} is not a download attachment")
    return f"HTTP {status}"


def no_temp_artifacts() -> str:
    artifacts = sorted(
        str(path.relative_to(ROOT))
        for folder in (ROOT / "convex", ROOT / "scripts", ROOT / "tests")
        for pattern in ("*.orig", "*.rej")
        for path in folder.rglob(pattern)
    )
    check(not artifacts, "temporary patch artifacts found: " + ", ".join(artifacts))
    return "no temporary patch artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="also check public URLs")
    arguments = parser.parse_args()
    started = time.monotonic()
    checks: list[str] = []
    try:
        run(["git", "diff", "--check"], "whitespace check")
        checks.append("git diff --check")
        checks.append(config_check())
        checks.append(deck_check())
        checks.append(no_temp_artifacts())
        run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", *FOCUSED_TESTS], "focused tests")
        checks.append("focused tests: dashboard, policy, retry, directional research")
        checks.append("local dashboard: " + http_check("http://127.0.0.1:8000/healthz"))
        if arguments.public:
            checks.append("public dashboard: " + http_check("https://convex.isobars.xyz/healthz"))
            checks.append(
                "public deck: "
                + http_check(
                    "https://convex.isobars.xyz/download/convex-deck",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            )
    except (CheckFailure, subprocess.TimeoutExpired) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        return 1

    for detail in checks:
        print(f"PASS  {detail}")
    print(f"READY  submission gate passed in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
