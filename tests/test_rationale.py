from convex.rationale import _has_only_brief_numbers


def test_narration_may_restate_but_not_change_a_deterministic_number():
    brief = "probability 0.600; net edge 12.30 dollars; size 2 contracts"
    assert _has_only_brief_numbers("The 0.600 signal leaves 12.30 net for 2 contracts.", brief)
    assert not _has_only_brief_numbers("The 0.600 signal leaves 12.31 net.", brief)
