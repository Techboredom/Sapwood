from sapwood.retrieval import BlockSelectionPolicy, recency_scorer


def _session(n_blocks=5, token_count=16):
    return {
        "session_id": "s",
        "model_id": "m",
        "blocks": [
            {"block_id": f"b{i}", "start_pos": i * 16, "token_count": token_count, "kind": "prefix"}
            for i in range(n_blocks)
        ],
    }


def test_recency_keeps_newest_first_in_window():
    pol = BlockSelectionPolicy(max_blocks=2)
    selected = pol.select(_session(5))
    # most recent 2 blocks (b3, b4)
    assert {b["block_id"] for b in selected} == {"b3", "b4"}
    # ordered ascending by start_pos
    assert [b["start_pos"] for b in selected] == sorted(b["start_pos"] for b in selected)


def test_pinned_blocks_are_kept():
    session = _session(5)
    pol = BlockSelectionPolicy(max_blocks=2, pinned=["b0"])
    selected = pol.select(session)
    ids = {b["block_id"] for b in selected}
    assert "b0" in ids  # pinned


def test_pinned_does_not_exceed_window():
    session = _session(6)
    pol = BlockSelectionPolicy(max_blocks=2, pinned=["b0"])
    selected = pol.select(session)
    assert "b0" in {b["block_id"] for b in selected}
    assert len(selected) <= 3  # pinned + 2 recency


def test_scoring_path():
    session = _session(5)
    pol = BlockSelectionPolicy(max_blocks=2, scorer=recency_scorer())
    selected = pol.select(session)
    assert len(selected) == 2
    # recency_scorer gives higher score to later blocks, so b4 and b3
    assert {b["block_id"] for b in selected} == {"b3", "b4"}


def test_recency_scorer_monotone_with_pos():
    scorer = recency_scorer()
    assert scorer({"start_pos": 100}) > scorer({"start_pos": 0})
