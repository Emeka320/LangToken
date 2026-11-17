from typing import Any

from langtoken.base import Tokenizer, TokenizerMetadata


class CharacterTokenizer(Tokenizer):
    """
    Character-level tokenizer.

    Treats each Unicode code point as an individual token.

    Features:
        - Character-based vocabulary construction
        - Optional lowercasing
        - Whitespace preservation
        - Unicode support
        - Standardized LangToken serialization
    """

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    META = TokenizerMetadata(
        name="CharacterTokenizer",
        family="Character-Level",
        version="1.0.0",
        description="Character-level tokenizer with Unicode support."
    )

    # --------------------------------------------------------------
    # Special tokens
    # --------------------------------------------------------------

    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"

    SPECIALS = (PAD, UNK, BOS, EOS)

    # --------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        lowercase: bool = False
    ):
        self.lowercase = lowercase

        if vocab is None:
            self.vocab = {
                token: idx
                for idx, token in enumerate(self.SPECIALS)
            }
        else:
            self.vocab = dict(vocab)

        self._validate_vocab()

    # --------------------------------------------------------------
    # Vocabulary validation
    # --------------------------------------------------------------

    def _validate_vocab(self) -> None:
        for token in self.SPECIALS:
            if token not in self.vocab:
                raise ValueError(
                    f"Vocabulary missing required special token {token!r}."
                )

        ids = list(self.vocab.values())

        if any(
            not isinstance(idx, int) or isinstance(idx, bool) or idx < 0
            for idx in ids
        ):
            raise ValueError(
                "Vocabulary IDs must be non-negative integers."
            )

        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary contains duplicate IDs.")

    # --------------------------------------------------------------
    # Special token IDs
    # --------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.UNK]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS]

    # --------------------------------------------------------------
    # Basic properties
    # --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"CharacterTokenizer("
            f"vocab_size={len(self)}, "
            f"lowercase={self.lowercase}, "
            f"unk_id={self.unk_id})"
        )

    # --------------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Split text into individual Unicode code points.

        Spaces, tabs, newlines and punctuation are preserved.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if self.lowercase:
            text = text.lower()

        return list(text)

    def encode(self, text: str) -> list[str]:
        """Encode text into individual character tokens."""

        return self.tokenize(text)

    # --------------------------------------------------------------
    # Vocabulary construction
    # --------------------------------------------------------------

    def train(
        self,
        corpus: str | list[str],
        min_freq: int = 1,
        reset: bool = True
    ) -> None:
        """
        Build character vocabulary from a corpus.

        Args:
            corpus:
                A string or list of strings.

            min_freq:
                Minimum character frequency required for inclusion.

            reset:
                Whether to rebuild the vocabulary from scratch.
        """

        if isinstance(corpus, str):
            texts = [corpus]
        elif isinstance(corpus, list):
            texts = corpus
        else:
            raise TypeError(
                "corpus must be a string or a list of strings."
            )

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("All corpus entries must be strings.")

        if not isinstance(min_freq, int) or isinstance(min_freq, bool) or min_freq < 1:
            raise ValueError("min_freq must be a positive integer.")

        from collections import Counter

        counter = Counter()

        for text in texts:
            for char in self.tokenize(text):
                if char not in self.SPECIALS:
                    counter[char] += 1

        if reset:
            self.vocab = {
                token: idx
                for idx, token in enumerate(self.SPECIALS)
            }

        next_id = max(self.vocab.values()) + 1

        # Deterministic vocabulary ordering.
        for char, frequency in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        ):
            if frequency >= min_freq and char not in self.vocab:
                self.vocab[char] = next_id
                next_id += 1

        self._validate_vocab()

    # --------------------------------------------------------------
    # Token <-> ID conversion
    # --------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the character-to-ID vocabulary."""

        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        """Convert a character token into its ID."""

        return self.vocab.get(token, self.unk_id)

    def id_to_tok(self, idx: int) -> str:
        """Convert an ID into its corresponding character."""

        inverse = {
            index: token
            for token, index in self.vocab.items()
        }

        return inverse.get(idx, self.UNK)

    # --------------------------------------------------------------
    # Decoding
    # --------------------------------------------------------------

    def decode(self, tokens: list[str]) -> str:
        """
        Reconstruct text from character tokens.

        Unlike word-level decoding, no spacing or punctuation
        reconstruction is necessary.
        """

        return "".join(tokens)

    # --------------------------------------------------------------
    # Serialization state
    # --------------------------------------------------------------

    def _get_state(self) -> dict[str, Any]:
        """Return algorithm-specific serialization state."""

        return {
            "vocab": self.vocab,
            "lowercase": self.lowercase
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> "CharacterTokenizer":
        """Restore a tokenizer from serialized state."""

        if "vocab" not in state:
            raise ValueError("Missing vocabulary in tokenizer state.")

        return cls(
            vocab={
                token: int(idx)
                for token, idx in state["vocab"].items()
            },
            lowercase=state.get("lowercase", False)
        )
