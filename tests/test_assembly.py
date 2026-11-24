from sapwood.retrieval import (
    assemble_window, window_summary, combined_score, token_jaccard, recency_bonus,
    BlockSelectionPolicy,
)


def _session(pos_tokens, token_count=8):
    """Build a session from a list of (start_pos, tokens) pairs."""
    blocks = []
    for pos, toks in pos_tokens:
        blocks.append({
            "block_id": f"b_{pos}",
            "start_pos": pos,
            "token_count": token_count,
            "kind": "prefix",
            "tokens": toks,
        })
    return {"session_id": "s", "model_id": "m", "blocks": blocks}


def test_token_jaccard():
    assert token_jaccard([1, 2, 3], [1, 2, 4]) > 0
    assert token_jaccard([1, 2, 3], []) == 0.0
    assert token_jaccard([], [1, 2]) == 0.0
    assert token_jaccard([1, 2, 3], [1, 2, 3]) == 1.0


def test_recency_bonus_monotone_and_bounded():
    assert 0.0 <= recency_bonus({"start_pos": 0}, ref_pos=100) <= 1.0
    assert recency_bonus({"start_pos": 100}, ref_pos=100) == 1.0
    assert recency_bonus({"start_pos": 1000}, ref_pos=100) == 1.0  # clamped


def test_combined_score_weights_relevance_over_recency():
    # block A: high lexical overlap, old; block B: low overlap, recent
    a = {"start_pos": 10, "tokens": [1, 2, 3, 4]}
    b = {"start_pos": 90, "tokens": [5]}
    q = [1, 2, 3, 4]
    sa = combined_score(a, q, ref_pos=100, weights=(1.0, 0.1))
    sb = combined_score(b, q, ref_pos=100, weights=(1.0, 0.1))
    assert sa > sb


def test_assemble_window_prefers_relevant_blocks():
    session = _session([
        (0,  [1, 2, 3, 4]),     # relevant, old
        (80, [99, 98, 97]),      # irrelevant, recent
        (160, [1, 2]),           # relevant, recent
    ])
    q = [1, 2, 3, 4]
    selected = assemble_window(session, q, max_blocks=2, anchor_blocks=None)
    ids = {b["block_id"] for b in selected}
    assert "b_0" in ids and "b_160" in ids  # the two relevant blocks
    assert "b_80" not in ids
    # ordered oldest-first by position
    assert [b["start_pos"] for b in selected] == sorted(b["start_pos"] for b in selected)


def test_assemble_window_pinned_anchors_always_included():
    session = _session([
        (0,  [1, 2, 3, 4]),     # pinned anchor (system prompt)
        (80, [9, 8, 7]),
    ])
    selected = assemble_window(session, [9, 8], max_blocks=1, pinned=["b_0"])
    ids = {b["block_id"] for b in selected}
    assert "b_0" in ids  # pinned even though low relevance to the query
    assert len(selected) == 2  # anchor + recency/relevance pick


def test_assemble_window_max_tokens_budget():
    session = _session([
        (0,  [1, 2, 3, 4]),
        (40, [5, 6, 7, 8]),
        (80, [9, 10, 11, 12]),
    ], token_count=8)
    # each block has 8 tokens; budget of 17 tokens -> room for 2 blocks
    selected = assemble_window(session, [1, 2, 3, 4], max_blocks=3, max_tokens=17)
    assert sum(b.get("token_count", 0) for b in selected) <= 17
    assert len(selected) <= 2


def test_assemble_window_without_query_falls_back_to_recency():
    session = _session([
        (0,  [1, 2, 3, 4]),
        (80, [9, 8, 7]),
        (160, [1, 2]),
    ])
    selected = assemble_window(session, None, max_blocks=2)
    # no query -> relevance 0; recency dominates -> most recent blocks
    assert selected  # non-empty
    assert selected[0]["start_pos"] <= selected[-1]["start_pos"]


def test_assemble_window_empty_session():
    assert assemble_window({"blocks": []}, [1, 2]) == []


def test_window_summary():
    session = _session([(0, [1]), (16, [2]), (32, [3])], token_count=16)
    selected = assemble_window(session, [1, 2], max_blocks=2)
    s = window_summary(selected)
    assert s["num_blocks"] == len(selected)
    assert s["total_tokens"] == sum(b["token_count"] for b in selected)
    assert s["span_end"] > s["span_start"]


def test_block_selection_policy_still_works():
    """The back-compat recency policy is unaffected by the assembly addition."""
    session = _session([(0, [1]), (16, [2]), (32, [3])], token_count=16)
    policy = BlockSelectionPolicy(max_blocks=2)
    selected = policy.select(session)
    assert len(selected) == 2
