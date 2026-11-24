from sapwood.cacheblend import compute_reuse_plan, reuse_plan_summary, ReusePlan


def _block(i, token_count=16, start_pos=None):
    return {
        "block_id": f"b{i}",
        "start_pos": start_pos if start_pos is not None else i * 16,
        "token_count": token_count,
        "kind": "prefix",
    }


def _window(block_ids, *, token_count=16, starts=None):
    """Build a window of blocks in position order."""
    out = []
    for i, bid in enumerate(block_ids):
        st = starts[i] if starts is not None else i * token_count
        out.append({"block_id": f"b{bid}", "start_pos": st, "token_count": token_count, "kind": "prefix"})
    return sorted(out, key=lambda b: b["start_pos"])


def test_contiguous_window_no_recompute():
    # blocks tile positions with no gaps -> all reusable
    selected = _window([0, 1, 2, 3])
    plan = compute_reuse_plan(selected, recompute_fraction=0.15)
    assert plan.recompute == []
    assert len(plan.reusable) == 4
    assert plan.total_window_tokens == 64
    assert plan.recompute_fraction == 0.0


def test_gap_marks_block_after_gap_stale():
    # window: b0 (pos 0), b1 (pos 48, gap of 32 tokens before it)
    selected = [_block(0, token_count=16, start_pos=0), _block(1, token_count=16, start_pos=48)]
    plan = compute_reuse_plan(selected, recompute_fraction=0.5)
    ids_recompute = {b["block_id"] for b in plan.recompute}
    assert "b1" in ids_recompute           # stale block is recomputed
    assert "b0" in {b["block_id"] for b in plan.reusable}  # base block reused
    assert plan.staleness["b1"] == 32       # gap = 48 - (0 + 16) = 32
    assert plan.staleness["b0"] == 0.0


def test_first_block_is_base_never_stale():
    # first kept block starts at pos 1000 (deep in conversation); no "previous"
    selected = [_block(5, token_count=16, start_pos=1000), _block(6, token_count=16, start_pos=1016)]
    plan = compute_reuse_plan(selected, recompute_fraction=0.5)
    assert plan.staleness["b5"] == 0.0
    assert plan.staleness["b6"] == 0.0  # contiguous with b5


def test_pinned_anchor_never_recomputed_even_with_gap():
    # b0 pinned at pos 0; b1 at pos 48 (gap) -> b1 recomputed, b0 stays
    selected = [_block(0, token_count=16, start_pos=0), _block(1, token_count=16, start_pos=48)]
    plan = compute_reuse_plan(selected, recompute_fraction=0.5, pinned=["b0"])
    assert "b0" in {b["block_id"] for b in plan.reusable}
    assert "b1" in {b["block_id"] for b in plan.recompute}


def test_first_stale_block_always_recomputed_even_over_budget():
    # tiny budget (2 tokens) but block size 16: the most-stale block is recomputed anyway
    selected = [_block(0, token_count=16, start_pos=0), _block(1, token_count=16, start_pos=48)]
    plan = compute_reuse_plan(selected, recompute_fraction=0.001)  # tiny budget
    assert len(plan.recompute) == 1
    assert plan.recompute[0]["block_id"] == "b1"


def test_budget_respects_fraction_with_multiple_stale():
    # 3 stale blocks each 16 tokens (64 total); budget 0.3*64=19 tokens
    selected = [
        _block(0, token_count=16, start_pos=0),
        _block(1, token_count=16, start_pos=48),   # staleness 32
        _block(2, token_count=16, start_pos=96),  # staleness 32
    ]
    plan = compute_reuse_plan(selected, recompute_fraction=0.3)
    # most-stale first; only b1 (16 tokens, 16<=19) fits; b2 would exceed -> reuse
    assert {b["block_id"] for b in plan.recompute} == {"b1"}
    assert {b["block_id"] for b in plan.reusable} == {"b0", "b2"}


def test_recompute_fraction_zero_disables_recompute():
    selected = [_block(0, token_count=16, start_pos=0), _block(1, token_count=16, start_pos=48)]
    plan = compute_reuse_plan(selected, recompute_fraction=0.0)
    assert plan.recompute == []
    assert len(plan.reusable) == 2


def test_custom_staleness_fn_is_used():
    selected = [_block(0, token_count=16, start_pos=0), _block(1, token_count=16, start_pos=48)]
    calls = []

    def fn(block, prev):
        calls.append(block["block_id"])
        return 42.0

    plan = compute_reuse_plan(selected, recompute_fraction=0.5, staleness_fn=fn)
    assert calls == ["b1"]               # only the non-first, non-pinned block uses the fn
    assert plan.staleness["b1"] == 42.0


def test_window_summary_and_dict_view():
    selected = [_block(0, token_count=16, start_pos=0), _block(1, token_count=16, start_pos=48)]
    plan = compute_reuse_plan(selected, recompute_fraction=0.5)
    summary = reuse_plan_summary(plan)
    assert summary["total_window_tokens"] == 32
    assert summary["recompute_tokens"] == 16
    assert summary["recompute_fraction"] == 0.5
    assert summary["num_recompute"] == 1

    d = plan.as_dict()
    assert d["recompute_ids"] == ["b1"]
    assert d["reusable_ids"] == ["b0"]


def test_recompute_blocks_sorted_by_start_pos():
    # three stale blocks, recompute all (large budget); order should be start_pos-sorted
    selected = [
        _block(0, token_count=16, start_pos=0),
        _block(1, token_count=16, start_pos=48),
        _block(2, token_count=16, start_pos=96),
    ]
    plan = compute_reuse_plan(selected, recompute_fraction=0.95)
    # staleness b1 == 32, b2 == 32; both recomputed; order preserved by start_pos
    assert [b["block_id"] for b in plan.recompute] == ["b1", "b2"]
    assert plan.reusable == [selected[0]]
