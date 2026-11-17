from __future__ import annotations

import re
from typing import Any

from langtoken.base import Tokenizer, TokenizerMetadata
from langtoken.subword.bpe import BPE
from langtoken.subword.unigram import Unigram


class SentencePiece(Tokenizer):
    """
    SentencePiece-style wrapper around LangToken's BPE and Unigram models.

    This is a SentencePiece-inspired tokenizer, not a claim of binary/model
    compatibility with Google's SentencePiece implementation.

    Text is converted to a continuous stream by replacing each run of
    whitespace with the visible word-boundary marker (default: ``▁``).
    An optional dummy prefix adds a marker at the start of non-empty input.
    """

    META = TokenizerMetadata(
        name="SentencePiece",
        family="subword",
        version="0.1.0",
        description=(
            "SentencePiece-style whitespace-marker wrapper using LangToken "
            "BPE or Unigram backends."
        ),
    )

    SUBTYPES = {"BPE", "Unigram"}

    def __init__(
        self,
        subtype: str = "Unigram",
        *,
        add_dummy_prefix: bool = True,
        marker: str = "▁",
        **backend_config: Any,
    ) -> None:
        if not isinstance(subtype, str):
            raise TypeError("subtype must be a string.")

        subtype = subtype.strip()
        if subtype not in self.SUBTYPES:
            raise ValueError(
                f"subtype must be one of {sorted(self.SUBTYPES)}."
            )

        if not isinstance(add_dummy_prefix, bool):
            raise TypeError("add_dummy_prefix must be a boolean.")

        if not isinstance(marker, str) or len(marker) != 1:
            raise ValueError("marker must be exactly one character.")

        if marker.isspace():
            raise ValueError("marker cannot be a whitespace character.")

        self.subtype = subtype
        self.add_dummy_prefix = add_dummy_prefix
        self.marker = marker
        self.backend_config = dict(backend_config)
        self.backend: BPE | Unigram | None = None

    def _normalise(self, text: str, *, add_prefix: bool) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if self.marker in text:
            raise ValueError(
                f"Input contains the reserved SentencePiece marker "
                f"{self.marker!r}. Choose another marker or remove it "
                f"before tokenization."
            )

        # SentencePiece-style whitespace normalisation: one marker per run.
        transformed = re.sub(r"\s+", self.marker, text)

        if add_prefix and transformed:
            transformed = self.marker + transformed

        return transformed

    def _require_backend(self) -> BPE | Unigram:
        if self.backend is None:
            raise RuntimeError("SentencePiece tokenizer has not been trained.")
        return self.backend

    def train(
        self,
        corpus: str | list[str],
        vocab_size: int = 8000,
        **kwargs: Any,
    ) -> None:
        """
        Train the selected backend on marker-normalised documents.

        BPE-specific training options (currently min_frequency) are passed
        to BPE.train. Unigram configuration can be supplied to the
        constructor; Unigram training overrides may be supplied here.
        """
        if isinstance(corpus, str):
            documents = [corpus]
        elif isinstance(corpus, list) and all(
            isinstance(document, str) for document in corpus
        ):
            documents = corpus
        else:
            raise TypeError("corpus must be a string or a list of strings.")

        if not documents or not any(documents):
            raise ValueError("Cannot train SentencePiece on an empty corpus.")

        processed = [
            self._normalise(
                document,
                add_prefix=self.add_dummy_prefix,
            )
            for document in documents
        ]

        if self.subtype == "BPE":
            allowed_config = set()
            unknown_config = set(self.backend_config) - allowed_config
            if unknown_config:
                raise TypeError(
                    "Unsupported BPE constructor options: "
                    f"{sorted(unknown_config)}"
                )

            backend = BPE(raw_stream=True)
            backend.train(
                processed,
                vocab_size=vocab_size,
                **kwargs,
            )

        else:
            config = dict(self.backend_config)
            config.setdefault("vocab_size", vocab_size)
            backend = Unigram(**config)
            backend.train(processed, **kwargs)

        self.backend = backend

    def encode(self, text: str) -> list[str]:
        backend = self._require_backend()
        processed = self._normalise(
            text,
            add_prefix=self.add_dummy_prefix,
        )
        return backend.encode(processed)

    def decode(self, tokens: list[str]) -> str:
        backend = self._require_backend()
        transformed = backend.decode(tokens)

        # Remove only the synthetic prefix. A marker remaining at the
        # beginning represents actual leading whitespace in the input.
        if self.add_dummy_prefix and transformed.startswith(self.marker):
            transformed = transformed[len(self.marker):]

        return transformed.replace(self.marker, " ")

    def get_vocab(self) -> dict[str, int]:
        return self._require_backend().get_vocab()

    def _get_state(self) -> dict[str, Any]:
        backend = self._require_backend()
        return {
            "subtype": self.subtype,
            "add_dummy_prefix": self.add_dummy_prefix,
            "marker": self.marker,
            "backend_config": self.backend_config,
            "backend_state": backend._get_state(),
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> "SentencePiece":
        if not isinstance(state, dict):
            raise ValueError("Invalid SentencePiece tokenizer state.")

        subtype = state.get("subtype")
        if subtype not in cls.SUBTYPES:
            raise ValueError("Invalid SentencePiece subtype.")

        instance = cls(
            subtype=subtype,
            add_dummy_prefix=state.get("add_dummy_prefix", True),
            marker=state.get("marker", "▁"),
            **state.get("backend_config", {}),
        )

        backend_state = state.get("backend_state")
        if not isinstance(backend_state, dict):
            raise ValueError("Missing or invalid backend state.")

        if subtype == "BPE":
            instance.backend = BPE._from_state(backend_state)
        else:
            instance.backend = Unigram._from_state(backend_state)

        return instance

    def __repr__(self) -> str:
        backend = (
            repr(self.backend)
            if self.backend is not None
            else "untrained"
        )
        return (
            f"SentencePiece(subtype={self.subtype!r}, "
            f"add_dummy_prefix={self.add_dummy_prefix}, "
            f"backend={backend})"
        )
