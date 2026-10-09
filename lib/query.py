"""Building the query text used for retrieval from a Case."""

from lib.dataset import Case
from lib.masking import mask_citations


def build_query(*, case: Case) -> str:
    """
    Build the query text for retrieval: `facts` and `considerations`, both with citation
    references masked out, concatenated (both fields are used, masked, not just
    `facts`).

    :param case: The case to build a query for.
    :returns: The masked, concatenated query text.
    """
    return f"{mask_citations(text=case.facts)} {mask_citations(text=case.considerations)}".strip()
