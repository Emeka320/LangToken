from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Any

from langtoken.base import Tokenizer, TokenizerMetadata


class BPE(Tokenizer):
    """
    Character-level BPE with frequency-based merges.

    The default mode retains the original whitespace/punctuation-aware
    pre-tokenization. Set raw_stream=True for continuous-stream BPE, where
    merges may cross spaces and punctuation (useful for SentencePiece-style
    tokenization).

    Pipeline:
      1. pre_tokenize(text) -> list of word chunks with trailing boundary
         markers (space or punctuation).
      2. Each chunk is represented as a tuple of symbols.
      3. Train merges over adjacent symbol pairs, weighted by word frequency.
      4. encode() replays the learned merges on each chunk.
      5. decode() simply concatenates chunk strings — no guessing.

    Boundary convention:
      - A trailing space " " marks "word followed by a space".
      - A punctuation character marks "word followed by that punctuation".
      - The last chunk has no trailing boundary.
      - This makes decode() lossless: "".join(chunks) == original text.

    LIMITATION: characters never seen during training map to <unk>, exactly
    as in the reference algorithm. Use ByteLevelBPE for arbitrary-input
    lossless encoding.
    """

    META = TokenizerMetadata(
        name="BPE",
        family="subword",
        version="2.0.0",
        description="Character-level BPE with whitespace + punctuation "
                    "pre-tokenization and lossless whitespace-preserving decode.",
    )

    PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
    SPECIALS = (PAD, UNK, BOS, EOS)

    # Punctuation that terminates a word chunk.
    _PUNCT = set(".,!?;:\"'()[]{}<>-—–/\\")
    _PUNCT_RE = re.compile(r"([" + re.escape("".join(_PUNCT)) + r"])")

    def __init__(self, raw_stream: bool = False):
        """
        Create a BPE tokenizer.

        raw_stream=False preserves the original whitespace/punctuation
        pre-tokenization behaviour.

        raw_stream=True treats each training document and each input string
        as one continuous character stream. No word or punctuation boundaries
        are introduced by the tokenizer.
        """
        if not isinstance(raw_stream, bool):
            raise TypeError("raw_stream must be a boolean.")

        self.raw_stream = raw_stream
        self._vocab: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._cache: dict[str, list[str]] = {}

    # ==================================================================
    # Pre-tokenization
    # ==================================================================
    @classmethod
    def pre_tokenize(cls, text: str) -> list[str]:
        """
        Split text into chunks preserving original spacing.

        Each chunk ends with either a space, a punctuation char, or nothing
        (if it's the final chunk). Concatenating the chunks reproduces text.

        Examples:
            "The mailman was"    -> ["The ", "mailman ", "was"]
            "dogs, just"         -> ["dogs", ", ", "just"]
            "street."            -> ["street", "."]
            "dogs ,"             -> ["dogs ", ","]   (space kept on prior chunk)
        """
        if not text:
            return []

        chunks: list[str] = []
        i = 0
        n = len(text)
        current: list[str] = []

        while i < n:
            ch = text[i]
            if ch == " ":
                # Space terminates the current word (if any).
                if current:
                    current.append(" ")
                    chunks.append("".join(current))
                    current = []
                else:
                    # Leading spaces attach to the *next* chunk by
                    # becoming part of it — preserve them verbatim.
                    current.append(" ")
                i += 1
            elif ch in cls._PUNCT:
                # Emit current word (if any), then punctuation as its own chunk.
                if current:
                    chunks.append("".join(current))
                    current = []
                chunks.append(ch)
                i += 1
            else:
                current.append(ch)
                i += 1

        if current:
            chunks.append("".join(current))
        return chunks

    # ==================================================================
    # Training
    # ==================================================================
    def train(
        self,
        corpus: list[str],
        vocab_size: int,
        *,
        min_frequency: int = 2,
        **kwargs,
    ) -> None:
        if not isinstance(corpus, list) or not all(
            isinstance(line, str) for line in corpus
        ):
            raise TypeError("corpus must be a list of strings.")

        if not isinstance(vocab_size, int) or vocab_size < len(self.SPECIALS):
            raise ValueError(
                f"vocab_size must be an integer of at least "
                f"{len(self.SPECIALS)}."
            )

        if not isinstance(min_frequency, int) or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer.")

        unknown_options = set(kwargs)
        if unknown_options:
            raise TypeError(
                f"Unexpected training arguments: {sorted(unknown_options)}"
            )

        # In raw-stream mode each complete document is one sequence.
        # In ordinary mode retain the existing boundary-aware chunks.
        word_freqs: Counter[str] = Counter()
        for line in corpus:
            chunks = [line] if self.raw_stream else self.pre_tokenize(line)
            for word in chunks:
                if word:
                    word_freqs[word] += 1

        if not word_freqs:
            raise ValueError("Cannot train BPE on an empty corpus.")

        # 2. Base vocab: every character in every word + specials.
        #    NOTE: the trailing boundary char (space/punct) is a *character*
        #    like any other, so it participates in merges. That's how
        #    "mailman " becomes a single token if frequent enough.
        chars = sorted({ch for word in word_freqs for ch in word})
        self._vocab = {tok: i for i, tok in enumerate(
            list(self.SPECIALS) + chars
        )}

        # 3. Represent each unique word as a tuple of single-char symbols.
        splits: dict[str, tuple[str, ...]] = {
            w: tuple(w) for w in word_freqs
        }

        # 4. Iteratively merge the most frequent adjacent pair.
        self._merges = []
        while len(self._vocab) < vocab_size:
            pair_freqs: Counter[tuple[str, str]] = Counter()
            for word, syms in splits.items():
                freq = word_freqs[word]
                for a, b in zip(syms, syms[1:]):
                    pair_freqs[(a, b)] += freq

            if not pair_freqs:
                break

            (best_pair, best_count), = pair_freqs.most_common(1)
            if best_count < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            splits = {
                w: self._apply_merge(s, best_pair, merged)
                for w, s in splits.items()
            }
            self._merges.append(best_pair)
            if merged not in self._vocab:
                self._vocab[merged] = len(self._vocab)

        self._merge_ranks = {p: i for i, p in enumerate(self._merges)}
        self._cache.clear()

    @staticmethod
    def _apply_merge(
        syms: tuple[str, ...], pair: tuple[str, str], merged: str
    ) -> tuple[str, ...]:
        out: list[str] = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        return tuple(out)

    # ==================================================================
    # Encoding
    # ==================================================================
    def encode(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text:
            return []

        chunks = [text] if self.raw_stream else self.pre_tokenize(text)

        tokens: list[str] = []
        for chunk in chunks:
            tokens.extend(self._encode_word(chunk))
        return tokens

    def _encode_word(self, word: str) -> list[str]:
        if word in self._cache:
            return self._cache[word]

        syms = list(word)
        while len(syms) > 1:
            candidates = [
                (self._merge_ranks[(a, b)], i)
                for i, (a, b) in enumerate(zip(syms, syms[1:]))
                if (a, b) in self._merge_ranks
            ]
            if not candidates:
                break
            _, i = min(candidates)
            syms[i : i + 2] = [syms[i] + syms[i + 1]]

        result = [s if s in self._vocab else self.UNK for s in syms]
        self._cache[word] = result
        return result

    def decode(self, tokens: list[str]) -> str:
        # Chunks carry their own boundaries, so plain concatenation is correct.
        return "".join(
            t for t in tokens
            if t not in (self.PAD, self.BOS, self.EOS)
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    # ==================================================================
    # IDs — defensive: never KeyError on stray tokens
    # ==================================================================
    def encode_ids(self, text: str) -> list[int]:
        vocab = self.get_vocab()
        unk_id = vocab[self.UNK]
        return [vocab.get(t, unk_id) for t in self.encode(text)]

    def decode_ids(self, ids: list[int]) -> str:
        inverse = {i: t for t, i in self.get_vocab().items()}
        unk_id = self.get_vocab()[self.UNK]
        return self.decode([inverse.get(i, self.UNK) for i in ids])

    # ==================================================================
    # Persistence
    # ==================================================================
    def _get_state(self) -> dict[str, Any]:
        return {
            "vocab": self._vocab,
            "merges": [list(p) for p in self._merges],
            "raw_stream": self.raw_stream,
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> "BPE":
        if not isinstance(state, dict):
            raise ValueError("Invalid BPE tokenizer state.")

        obj = cls(raw_stream=state.get("raw_stream", False))
        obj._vocab = dict(state["vocab"])
        obj._merges = [tuple(p) for p in state["merges"]]
        obj._merge_ranks = {p: i for i, p in enumerate(obj._merges)}
        return obj

    def __repr__(self) -> str:
        return (
            f"BPE(vocab_size={len(self._vocab)}, "
            f"merges={len(self._merges)})"
        )