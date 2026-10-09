"""The language codes used throughout this project's data (de/fr/it)."""

from enum import StrEnum


class Language(StrEnum):
    DE = "de"
    FR = "fr"
    IT = "it"

    @property
    def nltk_name(self) -> str:
        """This language's name as nltk's stemmer/stopwords corpora expect it, e.g.
        "german" for `Language.DE`."""
        return {
            Language.DE: "german",
            Language.FR: "french",
            Language.IT: "italian",
        }[self]
