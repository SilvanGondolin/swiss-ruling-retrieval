from lib.dense.embedding_model import chunk_for_model, tokenizer
from lib.rulings import iter_rulings, list_ruling_ids


def test_chunk_for_model_fit_the_model_window_and_cover_the_text():
    # a real ruling long enough to need several chunks (checked below)
    [ruling] = iter_rulings(ruling_ids=frozenset(list_ruling_ids()[:1]))
    text = ruling.considerations
    chunks = chunk_for_model(text=text, prefix="query: ")
    assert len(chunks) > 1
    assert all(
        len(tokenizer()(f"query: {chunk}").input_ids) <= tokenizer().model_max_length
        for chunk in chunks
    )
    reassembled = "".join(chunks)
    assert "".join(reassembled.split()) == "".join(text.split())
