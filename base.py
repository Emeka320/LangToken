
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, ClassVar, Self
from inspect import isabstract


@dataclass(frozen=True)
class TokenizerMetadata:
    """Metadata declared by each tokenizer algorithm."""

    name: str
    family: str
    version: str
    description: str = ""


class Tokenizer(ABC):
    """
    Base class for all LangToken tokenizers.

    Provides:
        - Common tokenizer interface
        - Token and token ID conversion
        - Standardized JSON serialization
        - Shared metadata and representation

    Individual algorithms control their tokenization logic
    and provide their own internal state management.
    """

    # --------------------------------------------------------------
    # Class configuration
    # --------------------------------------------------------------

    META: ClassVar[TokenizerMetadata]

    FORMAT: ClassVar[str] = "langtoken"
    FORMAT_VERSION: ClassVar[str] = "1"

    UNK: ClassVar[str] = "<UNK>"

    # --------------------------------------------------------------
    # Subclass validation
    # --------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Abstract intermediate classes may defer metadata.
        if not isabstract(cls):
            meta = cls.__dict__.get("META")

            if not isinstance(meta, TokenizerMetadata):
                raise TypeError(
                    f"{cls.__name__} must declare META "
                    "as a TokenizerMetadata instance."
                )

    # --------------------------------------------------------------
    # Core interface — algorithms implement these
    # --------------------------------------------------------------

    @abstractmethod
    def train(self, corpus: str | list[str], **kwargs) -> None:
        """Build the vocabulary and algorithm-specific state."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[str]:
        """Tokenize text into token strings."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        """Reconstruct text from token strings."""
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        """Return the token string-to-ID mapping."""
        raise NotImplementedError

    # --------------------------------------------------------------
    # Token ID conversion
    # --------------------------------------------------------------

    def encode_ids(self, text: str) -> list[int]:
        """Encode text directly into token IDs."""

        vocab = self.get_vocab()

        if self.UNK not in vocab:
            raise ValueError(
                "Cannot encode token IDs: "
                "the tokenizer has no UNK token."
            )

        unk_id = vocab[self.UNK]

        return [
            vocab.get(token, unk_id)
            for token in self.encode(text)
        ]

    def decode_ids(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""

        vocab = self.get_vocab()

        inverse = {
            index: token
            for token, index in vocab.items()
        }

        if self.UNK not in vocab:
            if any(index not in inverse for index in ids):
                raise ValueError(
                    "Cannot decode unknown token IDs: "
                    "the tokenizer has no UNK token."
                )

        return self.decode([
            inverse.get(index, self.UNK)
            for index in ids
        ])

    # --------------------------------------------------------------
    # Algorithm-provided internal state
    # --------------------------------------------------------------

    @abstractmethod
    def _get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of internal state."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_state(cls, state: dict[str, Any]) -> Self:
        """Reconstruct an instance from its serialized internal state."""
        raise NotImplementedError

    # --------------------------------------------------------------
    # Shared serialization
    # --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the tokenizer using LangToken's standardized JSON format."""

        payload = {
            "format": self.FORMAT,
            "format_version": self.FORMAT_VERSION,
            "metadata": asdict(self.META),
            "state": self._get_state(),
        }

        path = Path(path)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a tokenizer from a LangToken JSON file."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        # Validate file format.
        if payload.get("format") != cls.FORMAT:
            raise ValueError(
                f"Invalid file format: expected {cls.FORMAT!r}, "
                f"got {payload.get('format')!r}."
            )

        # Validate serialization version.
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version "
                f"{payload.get('format_version')!r}; "
                f"expected {cls.FORMAT_VERSION!r}."
            )

        # Validate algorithm metadata.
        meta = payload.get("metadata")

        if meta != asdict(cls.META):
            raise ValueError(
                f"Metadata mismatch: file declares {meta}, "
                f"class declares {asdict(cls.META)}."
            )

        # Restore algorithm-specific state.
        if "state" not in payload:
            raise ValueError("Missing tokenizer state in file.")

        return cls._from_state(payload["state"])

    # --------------------------------------------------------------
    # Dunder methods
    # --------------------------------------------------------------

    def __call__(self, text: str) -> list[str]:
        """Allow tokenizer instances to be called directly."""

        return self.encode(text)

    def __repr__(self) -> str:
        """Return a readable tokenizer representation."""

        return (
            f"{type(self).__name__}("
            f"family={self.META.family!r}, "
            f"version={self.META.version!r}, "
            f"vocab_size={len(self.get_vocab())})"
        )


