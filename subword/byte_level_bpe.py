from __future__ import annotations
from collections import Counter
from typing import Any

from langtoken.base import Tokenizer, TokenizerMetadata
from langtoken.subword.bpe import BPE


class ByteLevelBPE(BPE):
    """
    Byte-level BPE: operates on raw UTF-8 bytes, so every input is encodable
    and no <unk> fallback is ever needed for arbitrary text.

    Inheritance relationship with BPE:
      - Reuses: encode()'s text preprocessing hook, get_vocab(), save/load
        envelope from Tokenizer, and the META enforcement from Tokenizer.
      - Overrides: train(), _encode_word(), _to_symbols(), get_vocab(),
        _get_state(), _from_state(), and the special-token handling.
      - Ignores: BPE's `</w>` and `▁` markers entirely — byte sequences are
        self-delimiting on decode because we concatenate bytes before decoding.
    """

    META = TokenizerMetadata(
        name="ByteLevelBPE",
        family="subword",
        version="1.0.0",
        description="Byte-level BPE over raw UTF-8 bytes. No OOV tokens.",
    )

    # Match the reference layout: 4 specials, then 256 byte tokens, then merges.
    PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
    SPECIALS = (PAD, UNK, BOS, EOS)
    BYTE_OFFSET = len(SPECIALS)          # 4
    BYTE_COUNT = 256
    BASE_VOCAB_SIZE = BYTE_OFFSET + BYTE_COUNT   # 260

    def __init__(self, lowercase: bool = False):
        # Deliberately do NOT call BPE.__init__ — its state shape
        # (string vocab + string merges + </w>) doesn't apply here.
        self.lowercase = lowercase
        self._merges: list[tuple[int, int]] = []          # ordered ID pairs
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._id_to_bytes: dict[int, bytes] = {}
        self._vocab: dict[str, int] = {}                  # label -> id
        self._id_to_token: dict[int, str] = {}
        self._cache: dict[str, list[int]] = {}
        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocabulary construction (ID <-> bytes <-> label)
    # ------------------------------------------------------------------
    @staticmethod
    def _byte_label(byte: int) -> str:
        return f"<0x{byte:02X}>"

    @staticmethod
    def _merge_label(data: bytes) -> str:
        return f"<bpe:{data.hex()}>"

    def _build_vocab(self) -> None:
        self._vocab = {tok: i for i, tok in enumerate(self.SPECIALS)}
        self._id_to_bytes = {}

        for byte in range(self.BYTE_COUNT):
            tid = self.BYTE_OFFSET + byte
            self._id_to_bytes[tid] = bytes([byte])
            self._vocab[self._byte_label(byte)] = tid

        for i, (left, right) in enumerate(self._merges):
            tid = self.BASE_VOCAB_SIZE + i
            if left not in self._id_to_bytes or right not in self._id_to_bytes:
                raise ValueError(f"Invalid merge pair at index {i}: {(left, right)}")
            merged = self._id_to_bytes[left] + self._id_to_bytes[right]
            self._id_to_bytes[tid] = merged
            self._vocab[self._merge_label(merged)] = tid

        self._id_to_token = {i: t for t, i in self._vocab.items()}
        self._merge_ranks = {p: i for i, p in enumerate(self._merges)}

    # ------------------------------------------------------------------
    # Training — byte IDs in, merge ID pairs out
    # ------------------------------------------------------------------
    def train(
        self,
        corpus: list[str],
        vocab_size: int = 8000,
        *,
        min_frequency: int = 2,
        reset: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(corpus, str):
            raise TypeError("train() expects an iterable of strings.")
        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be >= {self.BASE_VOCAB_SIZE}.")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1.")

        if reset:
            self._merges = []
            self._build_vocab()

        sequences: list[list[int]] = []
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError("Corpus items must be strings.")
            if self.lowercase:
                text = text.lower()
            ids = [b + self.BYTE_OFFSET for b in text.encode("utf-8")]
            if ids:
                sequences.append(ids)

        max_merges = vocab_size - self.BASE_VOCAB_SIZE
        for _ in range(max_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            (best_pair, freq), = pair_counts.most_common(1)
            if freq < min_frequency:
                break

            new_id = self.BASE_VOCAB_SIZE + len(self._merges)
            left, right = best_pair
            for i, seq in enumerate(sequences):
                out, pos = [], 0
                while pos < len(seq):
                    if (pos < len(seq) - 1
                            and seq[pos] == left
                            and seq[pos + 1] == right):
                        out.append(new_id)
                        pos += 2
                    else:
                        out.append(seq[pos])
                        pos += 1
                sequences[i] = out
            self._merges.append(best_pair)

        self._build_vocab()
        self._cache.clear()

    # ------------------------------------------------------------------
    # Encoding — always yields known byte/merge IDs, never <unk>
    # ------------------------------------------------------------------
    def _to_symbols(self, word: str) -> tuple[str, ...]:
        # Not used by this subclass; present so any BPE code that calls it
        # fails loudly rather than silently doing the wrong thing.
        raise NotImplementedError("ByteLevelBPE operates on bytes, not symbols.")

    def _encode_word(self, word: str) -> list[str]:
        raise NotImplementedError("ByteLevelBPE operates on bytes, not words.")

    def _apply_merges(self, byte_ids: list[int]) -> list[int]:
        seq = list(byte_ids)
        for i, (left, right) in enumerate(self._merges):
            if len(seq) <= 1:
                break
            new_id = self.BASE_VOCAB_SIZE + i
            out, pos = [], 0
            while pos < len(seq):
                if (pos < len(seq) - 1
                        and seq[pos] == left
                        and seq[pos + 1] == right):
                    out.append(new_id)
                    pos += 2
                else:
                    out.append(seq[pos])
                    pos += 1
            seq = out
        return seq

    def encode(self, text: str, add_specials: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode() expects a string.")
        if self.lowercase:
            text = text.lower()
        byte_ids = [b + self.BYTE_OFFSET for b in text.encode("utf-8")]
        ids = self._apply_merges(byte_ids)
        if add_specials:
            ids = [self.BOS] + ids + [self.EOS]
        return ids

    def encode_ids(self, text: str) -> list[int]:  # base contract
        return self.encode(text, add_specials=False)

    # ------------------------------------------------------------------
    # Decoding — concatenate bytes, then decode UTF-8 once
    # ------------------------------------------------------------------
    def decode(self, tokens: list[str] | list[int], skip_specials: bool = True) -> str:
        # The base class signature is list[str] -> str; for ByteLevelBPE the
        # natural unit is IDs. Accept both: if strings, map back to IDs first.
        ids = tokens if (not tokens or isinstance(tokens[0], int)) else [
            self._vocab.get(t, self.UNK_ID) for t in tokens
        ]
        return self.decode_to_string(ids, skip_specials=skip_specials)

    def decode_ids(self, ids: list[int]) -> str:  # base contract
        return self.decode_to_string(ids, skip_specials=True)

    def decode_to_string(self, ids: list[int], skip_specials: bool = True) -> str:
        buf = bytearray()
        parts: list[str] = []

        def flush() -> None:
            if buf:
                parts.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for idx in ids:
            if 0 <= idx < len(self.SPECIALS):
                if skip_specials:
                    continue
                flush()
                parts.append(self.SPECIALS[idx])
            elif idx in self._id_to_bytes:
                buf.extend(self._id_to_bytes[idx])
            else:
                if not skip_specials:
                    flush()
                    parts.append(self.UNK)
        flush()
        return "".join(parts)

    # ------------------------------------------------------------------
    # Vocab view — base expects dict[str, int]
    # ------------------------------------------------------------------
    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int: return 0
    @property
    def unk_id(self) -> int: return 1
    @property
    def bos_id(self) -> int: return 2
    @property
    def eos_id(self) -> int: return 3

    # ------------------------------------------------------------------
    # State for save/load
    # ------------------------------------------------------------------
    def _get_state(self) -> dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "merges": [list(p) for p in self._merges],
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> "ByteLevelBPE":
        obj = cls(lowercase=state.get("lowercase", False))
        obj._merges = [tuple(p) for p in state["merges"]]
        obj._build_vocab()
        return obj

    def __repr__(self) -> str:
        return (
            f"ByteLevelBPE(vocab_size={self.vocab_size}, "
            f"merges={len(self._merges)}, lowercase={self.lowercase})"
        )