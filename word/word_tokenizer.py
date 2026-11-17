import re
from collections import Counter
from typing import Any

from langtoken.base import Tokenizer, TokenizerMetadata


class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer.

    Tokenizes text into complete words, numbers and punctuation.

    Features:
        - Frequency-based vocabulary construction
        - Configurable lowercasing
        - Unicode-aware word recognition
        - Contraction handling
        - Punctuation attachment during decoding
        - Compound word reconstruction
        - Standardized LangToken serialization
    """

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    META = TokenizerMetadata(
        name="WordTokenizer",
        family="Word-Level",
        version="1.0.0",
        description="Frequency-based word-level tokenizer."
    )

    # --------------------------------------------------------------
    # Special tokens
    # --------------------------------------------------------------

    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"

    SPECIALS = (PAD, UNK, BOS, EOS)

    # Override the base tokenizer's UNK convention.
    # LangToken's base class uses this attribute in encode_ids().
    # --------------------------------------------------------------
    # Tokenization patterns
    # --------------------------------------------------------------

    _PATTERN = re.compile(
        r"\[[A-Z]+\]"
        r"|[^\W\d_]+(?:['’][^\W\d_]+)?"
        r"|\d+(?:\.\d+)?"
        r"|[^\w\s]|_",
        re.UNICODE
    )

    _ATTACH_TO_PREV = set(".,;:!?%)]}…»”")
    _ATTACH_TO_NEXT = set("([{¿¡«“")
    _COMPOUND_JOINERS = set("-/")

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
            f"WordTokenizer("
            f"vocab_size={len(self)}, "
            f"lowercase={self.lowercase}, "
            f"unk_id={self.unk_id})"
        )

    # --------------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Split text into words, numbers and punctuation.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if self.lowercase:
            text = text.lower()

        return self._PATTERN.findall(text)

    def encode(self, text: str) -> list[str]:
        """
        Encode text into word-level tokens.

        Returns token strings rather than token IDs.
        """

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
        Build vocabulary from a text corpus.

        Args:
            corpus:
                A string or list of strings.

            min_freq:
                Minimum token frequency required for inclusion.

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

        if not isinstance(min_freq, int) or min_freq < 1:
            raise ValueError("min_freq must be a positive integer.")

        counter = Counter()

        for text in texts:
            for token in self.tokenize(text):
                if token not in self.SPECIALS:
                    counter[token] += 1

        if reset:
            self.vocab = {
                token: idx
                for idx, token in enumerate(self.SPECIALS)
            }

        next_id = max(self.vocab.values()) + 1

        # Deterministic vocabulary ordering:
        # Highest frequency first, alphabetical order on ties.
        for token, frequency in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        ):
            if frequency >= min_freq and token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

        self._validate_vocab()

    # --------------------------------------------------------------
    # Token <-> ID conversion
    # --------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the token-to-ID vocabulary."""

        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        """Convert a token into its corresponding ID."""

        return self.vocab.get(token, self.unk_id)

    def id_to_tok(self, idx: int) -> str:
        """Convert an ID into its corresponding token."""

        inverse = {
            index: token
            for token, index in self.vocab.items()
        }

        return inverse.get(idx, self.UNK)

    # --------------------------------------------------------------
    # Token decoding
    # --------------------------------------------------------------

    def decode(self, tokens: list[str]) -> str:
        """
        Reconstruct readable text from word-level tokens.

        Handles punctuation attachment and compound joiners.
        """

        if not tokens:
            return ""

        output = [tokens[0]]

        for token in tokens[1:]:
            previous = output[-1]

            attach = (
                token in self._ATTACH_TO_PREV
                or previous in self._ATTACH_TO_NEXT
                or token in self._COMPOUND_JOINERS
                or previous.endswith(tuple(self._COMPOUND_JOINERS))
            )

            if attach:
                output[-1] = previous + token
            else:
                output.append(token)

        return " ".join(output)

    def decode_to_string(self, tokens: list[str]) -> str:
        """Alias for readable token reconstruction."""

        return self.decode(tokens)

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
    def _from_state(cls, state: dict[str, Any]) -> "WordTokenizer":
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
