from typing import Any

from langtoken.base import Tokenizer, TokenizerMetadata


class CharacterShingle(Tokenizer):
    """
    Character Shingle tokenizer.

    Splits text into overlapping character n-grams.

    Features:
        - Configurable shingle size
        - Overlapping character n-grams
        - Unicode support
        - Whitespace preservation
        - Optional lowercasing
        - Frequency-based vocabulary construction
        - Standardized LangToken serialization
    """

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    META = TokenizerMetadata(
        name="CharacterShingle",
        family="Character-Level",
        version="1.0.0",
        description="Character-level tokenizer using overlapping n-grams."
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
        n: int = 3,
        vocab: dict[str, int] | None = None,
        lowercase: bool = False
    ):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer.")

        self.n = n
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
            f"CharacterShingle("
            f"n={self.n}, "
            f"vocab_size={len(self)}, "
            f"lowercase={self.lowercase}, "
            f"unk_id={self.unk_id})"
        )

    # --------------------------------------------------------------
    # Character preprocessing
    # --------------------------------------------------------------

    def _prepare(self, text: str) -> str:
        """Apply configured text preprocessing."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if self.lowercase:
            text = text.lower()

        return text

    # --------------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Split text into overlapping character n-grams.

        Each consecutive window contains n characters.

        Example:
            n=3
            "LANGTOKEN" ->
            ["LAN", "ANG", "NGT", "GTO", "TOK", "OKE", "KEN"]
        """

        text = self._prepare(text)

        if len(text) < self.n:
            return []

        return [
            text[i:i + self.n]
            for i in range(len(text) - self.n + 1)
        ]

    def encode(self, text: str) -> list[str]:
        """Encode text into overlapping character shingles."""

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
        Build a vocabulary from character shingles.

        Args:
            corpus:
                A string or list of strings.

            min_freq:
                Minimum shingle frequency required for inclusion.

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

        if (
            not isinstance(min_freq, int)
            or isinstance(min_freq, bool)
            or min_freq < 1
        ):
            raise ValueError("min_freq must be a positive integer.")

        from collections import Counter

        counter = Counter()

        for text in texts:
            for shingle in self.tokenize(text):
                if shingle not in self.SPECIALS:
                    counter[shingle] += 1

        if reset:
            self.vocab = {
                token: idx
                for idx, token in enumerate(self.SPECIALS)
            }

        next_id = max(self.vocab.values()) + 1

        for shingle, frequency in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        ):
            if frequency >= min_freq and shingle not in self.vocab:
                self.vocab[shingle] = next_id
                next_id += 1

        self._validate_vocab()

    # --------------------------------------------------------------
    # Token <-> ID conversion
    # --------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the shingle-to-ID vocabulary."""

        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        """Convert a shingle into its corresponding ID."""

        return self.vocab.get(token, self.unk_id)

    def id_to_tok(self, idx: int) -> str:
        """Convert an ID into its corresponding shingle."""

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
        Reconstruct text from overlapping shingles.

        The first shingle is retained completely.
        Subsequent shingles contribute only their final character.

        Example:
            ["LAN", "ANG", "NGT"] -> "LANGT"
        """

        if not tokens:
            return ""

        if len(tokens) == 1:
            return tokens[0]

        result = tokens[0]

        for token in tokens[1:]:
            if len(token) != self.n:
                raise ValueError(
                    f"Invalid shingle {token!r}: "
                    f"expected length {self.n}."
                )

            if result[-(self.n - 1):] != token[:-1] if self.n > 1 else False:
                raise ValueError(
                    f"Non-overlapping shingles encountered: "
                    f"{result[-(self.n - 1):]!r} and {token!r}."
                )

            result += token[-1]

        return result

    # --------------------------------------------------------------
    # Serialization state
    # --------------------------------------------------------------

    def _get_state(self) -> dict[str, Any]:
        """Return algorithm-specific serialization state."""

        return {
            "n": self.n,
            "vocab": self.vocab,
            "lowercase": self.lowercase
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> "CharacterShingle":
        """Restore a tokenizer from serialized state."""

        if "vocab" not in state:
            raise ValueError("Missing vocabulary in tokenizer state.")

        return cls(
            n=state.get("n", 3),
            vocab={
                token: int(idx)
                for token, idx in state["vocab"].items()
            },
            lowercase=state.get("lowercase", False)
        )
