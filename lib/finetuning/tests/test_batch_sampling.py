from lib.finetuning.batch_sampling import distinct_batches


def _check_distinct(*, batches, case_ids, ruling_ids):
    for batch in batches:
        assert len({case_ids[i] for i in batch}) == len(batch)
        assert len({ruling_ids[i] for i in batch}) == len(batch)


def test_batches_never_repeat_a_case_or_a_ruling():
    # 40 pairs: 10 cases x 4 rulings each, rulings shared across cases
    case_ids = [f"case{i // 4}" for i in range(40)]
    ruling_ids = [f"ruling{(i * 7) % 13}" for i in range(40)]
    batches = distinct_batches(
        case_ids=case_ids, ruling_ids=ruling_ids, batch_size=5, seed=0
    )
    assert batches
    _check_distinct(batches=batches, case_ids=case_ids, ruling_ids=ruling_ids)


def test_every_batch_is_full_and_no_pair_is_used_twice():
    case_ids = [f"case{i}" for i in range(50)]
    ruling_ids = [f"ruling{i}" for i in range(50)]
    batches = distinct_batches(
        case_ids=case_ids, ruling_ids=ruling_ids, batch_size=8, seed=1
    )
    used = [i for batch in batches for i in batch]
    assert all(len(batch) == 8 for batch in batches)
    assert len(used) == len(set(used)) == 48  # 6 full batches, 2 pairs dropped


def test_a_ruling_cited_more_often_than_there_are_batches_is_partly_dropped():
    # 3 pairs all cite "popular": at most one fits per batch, and with batch_size 2
    # only 2 batches can be formed from the 4 other pairs + the popular one
    case_ids = ["a", "b", "c", "d", "e", "f"]
    ruling_ids = ["popular", "popular", "popular", "x", "y", "z"]
    batches = distinct_batches(
        case_ids=case_ids, ruling_ids=ruling_ids, batch_size=2, seed=0
    )
    _check_distinct(batches=batches, case_ids=case_ids, ruling_ids=ruling_ids)
    used = [i for batch in batches for i in batch]
    assert sum(ruling_ids[i] == "popular" for i in used) <= len(batches)


def test_different_seeds_give_different_orders_and_same_seed_repeats():
    case_ids = [f"case{i}" for i in range(30)]
    ruling_ids = [f"ruling{i}" for i in range(30)]
    first = distinct_batches(
        case_ids=case_ids, ruling_ids=ruling_ids, batch_size=5, seed=0
    )
    assert first == distinct_batches(
        case_ids=case_ids, ruling_ids=ruling_ids, batch_size=5, seed=0
    )
    assert first != distinct_batches(
        case_ids=case_ids, ruling_ids=ruling_ids, batch_size=5, seed=1
    )


def test_fewer_pairs_than_a_batch_gives_no_batches():
    assert (
        distinct_batches(case_ids=["a"], ruling_ids=["x"], batch_size=2, seed=0) == []
    )
