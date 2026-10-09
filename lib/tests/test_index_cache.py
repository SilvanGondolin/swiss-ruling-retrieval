from lib.index_cache import cache_name


def test_full_corpus_uses_the_plain_base_name():
    assert cache_name(base="dense_index", ruling_ids=None) == "dense_index"


def test_subset_name_contains_the_base_and_the_ruling_count():
    name = cache_name(base="dense_index", ruling_ids=frozenset({"a", "b", "c"}))
    assert name.startswith("dense_index_n3_")


def test_same_rulings_give_the_same_name_regardless_of_order():
    assert cache_name(base="x", ruling_ids=frozenset({"a", "b"})) == cache_name(
        base="x", ruling_ids=frozenset({"b", "a"})
    )


def test_different_rulings_of_the_same_size_give_different_names():
    assert cache_name(base="x", ruling_ids=frozenset({"a", "b"})) != cache_name(
        base="x", ruling_ids=frozenset({"a", "c"})
    )


def test_different_bases_give_different_names():
    ids = frozenset({"a"})
    assert cache_name(base="bm25_index", ruling_ids=ids) != cache_name(
        base="dense_index", ruling_ids=ids
    )
