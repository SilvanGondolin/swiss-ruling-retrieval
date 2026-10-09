from lib.sampling import sample_items


def test_sample_items_is_reproducible_and_keeps_original_order():
    items = list(range(100))
    first = sample_items(items=items, n=10, seed=0)
    assert first == sample_items(items=items, n=10, seed=0)
    assert first == sorted(first)
    assert len(set(first)) == 10


def test_sample_items_differs_between_seeds():
    items = list(range(100))
    assert sample_items(items=items, n=10, seed=0) != sample_items(
        items=items, n=10, seed=1
    )


def test_sample_items_returns_everything_when_n_exceeds_population():
    assert sample_items(items=[1, 2, 3], n=10, seed=0) == [1, 2, 3]


def test_sample_items_returns_everything_when_n_is_none():
    assert sample_items(items=[1, 2, 3], n=None, seed=0) == [1, 2, 3]
