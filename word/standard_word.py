from collections import Counter
from typing import Self

import regex

from langtoken.base import Tokenizer, TokenizerMetadata


class StandardWordTokenizer(Tokenizer):
    """
    Unicode Standard word tokenizer.

    Uses Unicode Text Segmentation (UAX #29) to identify
    word boundaries.

    Punctuation-only segments are discarded.

    Example:
        "The Brown-Foxes!"
        -> ["The", "Brown", "Foxes"]
    """

    META = TokenizerMetadata(
        name="StandardWordTokenizer",
        family="Word-Level",
        version="1.0.0",
        description="Word tokenizer using Unicode Text Segmentation."
    )

    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"

    SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)

    _WORD_CONTENT = regex.compile(
        r"[\p{L}\p{M}\p{N}]",
        regex.VERSION1,
    )

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        lowercase: bool = False,
    ):
        self.lowercase = lowercase

        if not isinstance(lowercase, bool):
            raise TypeError("lowercase must be a boolean.")

        if vocab is None:
            self.vocab = {
                token: index
                for index, token in enumerate(self.SPECIAL_TOKENS)
            }
        else:
            self.vocab = dict(vocab)

        self._validate_vocab()

    def _validate_vocab(self) -> None:
        for token in self.SPECIAL_TOKENS:
            if token not in self.vocab:
                raise ValueError(
                    f"Vocabulary is missing special token {token!r}."
                )

        ids = list(self.vocab.values())

        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in ids
        ):
            raise ValueError(
                "Vocabulary IDs must be non-negative integers."
            )

        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary IDs must be unique.")

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if self.lowercase:
            text = text.lower()

        if not text:
            return []

        # Locate Unicode word boundaries.
        boundaries = [
            match.start()
            for match in regex.finditer(
                r"\b",
                text,
                flags=regex.WORD | regex.VERSION1,
            )
        ]

        positions = sorted(set([0, *boundaries, len(text)]))

        tokens = []

        for start, end in zip(positions, positions[1:]):
            segment = text[start:end]

            # Keep segments containing letters, marks or numbers.
            # Discard whitespace and punctuation-only segments.
            if self._WORD_CONTENT.search(segment):
                tokens.append(segment)

        return tokens

    def train(
        self,
        corpus: str | list[str],
        min_freq: int = 1,
        reset: bool = True,
    ) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]

        if not isinstance(corpus, list) or not all(
            isinstance(text, str) for text in corpus
        ):
            raise TypeError("corpus must be a string or list of strings.")

        if (
            isinstance(min_freq, bool)
            or not isinstance(min_freq, int)
            or min_freq < 1
        ):
            raise ValueError("min_freq must be a positive integer.")

        if not isinstance(reset, bool):
            raise TypeError("reset must be a boolean.")

        if reset:
            self.vocab = {
                token: index
                for index, token in enumerate(self.SPECIAL_TOKENS)
            }

        frequencies = Counter()

        for text in corpus:
            frequencies.update(self.tokenize(text))

        sorted_tokens = sorted(
            (
                (token, frequency)
                for token, frequency in frequencies.items()
                if frequency >= min_freq
                and token not in self.SPECIAL_TOKENS
                and token not in self.vocab
            ),
            key=lambda item: (-item[1], item[0]),
        )

        for token, _ in sorted_tokens:
            self.vocab[token] = len(self.vocab)

    def encode(self, text: str) -> list[str]:
        return self.tokenize(text)

    def decode(self, tokens: list[str]) -> str:
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list of strings.")

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Every token must be a string.")

        return " ".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.UNK])

    def id_to_tok(self, index: int) -> str:
        inverse = {
            value: key
            for key, value in self.vocab.items()
        }

        return inverse.get(index, self.UNK)

    def _get_state(self) -> dict:
        return {
            "vocab": self.vocab,
            "lowercase": self.lowercase,
        }

    @classmethod
    def _from_state(cls, state: dict) -> Self:
        return cls(
            vocab=state["vocab"],
            lowercase=state["lowercase"],
        )
