from convex.rationale import _has_only_brief_numbers


def test_narration_may_restate_but_not_change_a_deterministic_number():
    brief = "probability 0.600; net edge 12.30 dollars; size 2 contracts"
    assert _has_only_brief_numbers("The 0.600 signal leaves 12.30 net for 2 contracts.", brief)
    assert not _has_only_brief_numbers("The 0.600 signal leaves 12.31 net.", brief)


def test_narration_may_not_convert_a_decimal_into_a_percentage():
    # The live failure. Phi-3 rewrote 0.600 as 60%, which is the same quantity
    # in different clothes and still a number the deterministic core did not
    # write. Every Featherless call was being discarded on this.
    brief = "probability 0.600; net edge 12.30 dollars"
    assert not _has_only_brief_numbers("A 60% chance leaves 12.30 net.", brief)
    assert _has_only_brief_numbers("A 0.600 chance leaves 12.30 net.", brief)


def test_a_split_contract_symbol_is_not_read_as_an_invented_number():
    # An OCC symbol is mostly digits. A model that breaks one across a phrase
    # emits a run of them that could never have been in the brief's number set,
    # because in the brief they were welded to letters. That is a mangled
    # identifier rather than an invented quantity, and retiring the narration
    # layer over it would be a defect in this check, not in the narration.
    brief = "widest leg is spy260831c00774000 at 28.6% of mid against a 10.5% limit"
    assert _has_only_brief_numbers(
        "The leg 260831c00774 sits at 28.6% of mid against a 10.5% limit.", brief
    )
    # The quantities themselves are still held to the letter.
    assert not _has_only_brief_numbers(
        "The leg spy260831c00774000 sits at 29% of mid against a 10.5% limit.", brief
    )
