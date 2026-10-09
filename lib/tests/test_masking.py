from lib.masking import Citation, find_citations, mask_citations


def test_masks_ref_law_tag():
    assert mask_citations(text="Recht von Amtes wegen an (<ref-law>).") == (
        "Recht von Amtes wegen an ([LAW])."
    )


def test_masks_ref_ruling_tag():
    assert mask_citations(text="siehe <ref-ruling> dazu") == "siehe [RULING] dazu"


def test_masks_two_chained_article_citations():
    # real example: case 001027b9-e219-4e03-9691-535a25cf3dad, `considerations`
    text = "dass die Beschwerde nicht innert der nach Art. 100 Abs. 1 BGG 30-tägigen, gemäss Art. 44 - 48 BGG am 17. Februar 2016 abgelaufenen Rechtsmittelfrist eingereicht worden ist,"
    masked = mask_citations(text=text)
    assert "Art." not in masked
    assert "BGG" not in masked
    assert masked.count("[LAW]") == 2


def test_masks_article_with_dash_range_qualifier():
    # real example: case 048eb2ad-2de4-4b56-9739-e246e4490c5e, `considerations`
    text = "Eine Parteientschädigung ist nicht zuzusprechen (Art. 68 Abs. 1-3 BGG)."
    assert (
        mask_citations(text=text)
        == "Eine Parteientschädigung ist nicht zuzusprechen ([LAW])."
    )


def test_masks_two_articles_joined_by_und():
    # real example: case 048eb2ad-2de4-4b56-9739-e246e4490c5e, `facts`
    text = "Ein Schriftenwechsel wurde nicht angeordnet (Art. 129 Abs. 3 und Art. 127 BGG)."
    masked = mask_citations(text=text)
    assert "Art." not in masked
    assert "BGG" not in masked
    assert masked.count("[LAW]") == 2


def test_french_article_citation():
    assert mask_citations(text="art. 86 al. 1 let. d LTF) par l'intéressé") == (
        "[LAW]) par l'intéressé"
    )


def test_bge_ruling_citation_number_is_masked():
    text = "impliquant une pesée des intérêts; ATF 1C_86/2015 du 20 avril 2016 consid. 3.5.1"
    masked = mask_citations(text=text)
    assert "1C_86/2015" not in masked
    assert "[RULING]" in masked


def test_sr_statute_number_is_masked():
    assert mask_citations(
        text="geregelt nach SR 173.110 und weiteren Bestimmungen."
    ) == ("geregelt nach [LAW] und weiteren Bestimmungen.")


def test_known_gap_chained_letter_list_partially_leaks():
    # Documented limitation (see lib/masking.py docstring): "et b LTF" is not
    # absorbed into the first citation's mask. This test pins the current,
    # known-imperfect behaviour rather than asserting a stronger guarantee we
    # don't actually provide.
    text = "art. 108 al. 1 let. a et b LTF."
    masked = mask_citations(text=text)
    assert masked == "[LAW] et b LTF."


def test_ordinary_text_is_unchanged():
    text = "Die Beschwerde wird abgewiesen, soweit darauf einzutreten ist."
    assert mask_citations(text=text) == text


def test_empty_string():
    assert mask_citations(text="") == ""


def test_find_citations_returns_matched_spans_with_labels():
    # tag pass runs before the article pass, so <ref-ruling> comes first
    text = "gemäss Art. 95 und Art. 96 BGG erhoben werden (<ref-ruling>)."
    assert find_citations(text=text) == [
        Citation(text="<ref-ruling>", label="RULING"),
        Citation(text="Art. 95", label="LAW"),
        Citation(text="Art. 96 BGG", label="LAW"),
    ]


def test_find_citations_matches_what_mask_citations_replaces():
    text = "art. 108 al. 1 let. a et b LTF. Weiteres SR 173.110 und ATF 1C_86/2015."
    for citation in find_citations(text=text):
        assert citation.text not in mask_citations(text=text)
