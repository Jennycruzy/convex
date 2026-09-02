"""The sequence is a receipt number, so two records may never share one.

The live ledger carries a scar from before this held: on 31 August the guard
issued 50 for a line the calibration job had already used, then ran one behind
until it was restarted, which is why seq 50 appears twice and 55 never appears
at all. The counter was cached when the Ledger was constructed and guarded by a
threading.Lock, which protects one process against itself and says nothing to
the four other processes writing the same file. It is now recounted inside an
advisory file lock. This is the test that would have caught it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

WRITER = textwrap.dedent(
    """
    import sys
    from convex.ledger import Action, Ledger, Record

    # Constructed up front, exactly as a long-lived guard process does, so the
    # cached counter is stale by the time anything is written.
    ledger = Ledger(sys.argv[1])
    for index in range(int(sys.argv[2])):
        ledger.append(
            Record(action=Action.SNAPSHOT, cycle_id="concurrency", rationale=f"{sys.argv[3]}-{index}")
        )
    """
)


def test_concurrent_processes_never_share_or_skip_a_sequence(tmp_path):
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    path = tmp_path / "decisions.jsonl"
    path.touch()

    writers = [
        subprocess.Popen(
            [sys.executable, str(script), str(path), "25", f"p{index}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(4)
    ]
    for writer in writers:
        _, stderr = writer.communicate(timeout=120)
        assert writer.returncode == 0, stderr.decode()

    sequences = [json.loads(line)["seq"] for line in path.read_text().splitlines() if line.strip()]
    assert len(sequences) == 100
    # Contiguous from one, every number used exactly once. A duplicate means two
    # writers agreed on a receipt number; a gap means one was issued and lost.
    assert sorted(sequences) == list(range(1, 101))
