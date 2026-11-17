from __future__ import annotations

import re
from collections import Counter
from typing import Any, ClassVar

from langtoken.base import Tokenizer, TokenizerMetadata


class WordPiece(Tokenizer):
    """WordPiece tokenizer trained with likelihood-style pair association scores.

    At each training iteration, adjacent pieces are scored using
    frequency(pair) / (frequency(left) * frequency(right)). Encoding uses
    greedy longest-match-first with ``##`` continuation pieces.

    Whitespace is retained as literal character tokens, allowing exact
    whitespace reconstruction when all input characters are in the vocabulary.
    """

    META: ClassVar[TokenizerMetadata] = TokenizerMetadata(
        name="WordPiece",
        family="subword",
        version="1.0.0",
        description=(
            "WordPiece tokenizer using likelihood-style pair scoring "
            "and greedy longest-match encoding."
        ),
    )

    PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
    SPECIALS = (PAD, UNK, BOS, EOS)

    def __init__(
        self,
        vocab_size: int = 8000,
        *,
        min_frequency: int = 2,
        continuation_prefix: str = "##",
    ) -> None:
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError("vocab_size must be an integer.")
        if vocab_size < len(self.SPECIALS) + 1:
            raise ValueError(
                f"vocab_size must be at least {len(self.SPECIALS) + 1}."
            )
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
            raise TypeError("min_frequency must be an integer.")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive.")
        if not isinstance(continuation_prefix, str) or not continuation_prefix:
            raise ValueError("continuation_prefix must be a non-empty string.")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.continuation_prefix = continuation_prefix
        self._vocab: dict[str, int] = {}
        self._merges: list[tuple[str, str, str]] = []
        self._trained = False

    @staticmethod
    def _split_corpus(corpus: str | list[str]) -> list[str]:
        if isinstance(corpus, str):
            documents = [corpus]
        elif isinstance(corpus, list) and all(
            isinstance(document, str) for document in corpus
        ):
            documents = corpus
        else:
            raise TypeError("corpus must be a string or a list of strings.")

        if not documents or not any(documents):
            raise ValueError("Cannot train on an empty corpus.")
        return documents

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        """Split text into word, punctuation, and whitespace segments."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        return re.findall(r"\w+|[^\w\s]+|\s+", text, flags=re.UNICODE)

    def _initial_pieces(self, segment: str) -> tuple[str, ...]:
        if segment.isspace():
            return tuple(segment)
        if not segment:
            return ()
        return (
            segment[0],
            *(self.continuation_prefix + char for char in segment[1:]),
        )

    def _merged_token(self, left: str, right: str) -> str:
        """Join two pieces without duplicating the continuation prefix."""
        if right.startswith(self.continuation_prefix):
            suffix = right[len(self.continuation_prefix):]
        else:
            suffix = right
        return left + suffix

    @staticmethod
    def _apply_merge(
        pieces: tuple[str, ...],
        pair: tuple[str, str],
        merged: str,
    ) -> tuple[str, ...]:
        output: list[str] = []
        index = 0
        while index < len(pieces):
            if (
                index + 1 < len(pieces)
                and pieces[index] == pair[0]
                and pieces[index + 1] == pair[1]
            ):
                output.append(merged)
                index += 2
            else:
                output.append(pieces[index])
                index += 1
        return tuple(output)

    def _pair_statistics(
        self,
        splits: dict[str, tuple[str, ...]],
        segment_frequencies: Counter[str],
    ) -> tuple[Counter[tuple[str, str]], Counter[str]]:
        pair_frequencies: Counter[tuple[str, str]] = Counter()
        token_frequencies: Counter[str] = Counter()

        for segment, pieces in splits.items():
            frequency = segment_frequencies[segment]
            for piece, count in Counter(pieces).items():
                token_frequencies[piece] += count * frequency
            for left, right in zip(pieces, pieces[1:]):
                pair_frequencies[(left, right)] += frequency

        return pair_frequencies, token_frequencies

    @staticmethod
    def _pair_score(
        pair_frequency: int,
        left_frequency: int,
        right_frequency: int,
    ) -> float:
        if pair_frequency <= 0 or left_frequency <= 0 or right_frequency <= 0:
            return 0.0
        return pair_frequency / (left_frequency * right_frequency)

    def train(
        self,
        corpus: str | list[str],
        vocab_size: int | None = None,
        *,
        min_frequency: int | None = None,
    ) -> None:
        """Learn pair merges until the target size or stopping condition."""
        documents = self._split_corpus(corpus)
        target_size = self.vocab_size if vocab_size is None else vocab_size
        minimum = self.min_frequency if min_frequency is None else min_frequency

        if not isinstance(target_size, int) or isinstance(target_size, bool):
            raise TypeError("vocab_size must be an integer.")
        if target_size < len(self.SPECIALS) + 1:
            raise ValueError(
                f"vocab_size must be at least {len(self.SPECIALS) + 1}."
            )
        if not isinstance(minimum, int) or isinstance(minimum, bool):
            raise TypeError("min_frequency must be an integer.")
        if minimum < 1:
            raise ValueError("min_frequency must be positive.")

        segment_frequencies: Counter[str] = Counter()
        for document in documents:
            segment_frequencies.update(self.pre_tokenize(document))

        splits = {
            segment: self._initial_pieces(segment)
            for segment in segment_frequencies
        }

        vocabulary = set(self.SPECIALS)
        for pieces in splits.values():
            vocabulary.update(pieces)

        if len(vocabulary) > target_size:
            raise ValueError(
                "vocab_size is smaller than the required special and "
                f"initial-character vocabulary ({len(vocabulary)} tokens required)."
            )

        merges: list[tuple[str, str, str]] = []
        while len(vocabulary) < target_size:
            pair_frequencies, token_frequencies = self._pair_statistics(
                splits, segment_frequencies
            )
            eligible = [
                pair for pair, count in pair_frequencies.items()
                if count >= minimum
            ]
            if not eligible:
                break

            # Deterministic tie-break: score, raw pair count, then pair tuple.
            best_pair = max(
                eligible,
                key=lambda pair: (
                    self._pair_score(
                        pair_frequencies[pair],
                        token_frequencies[pair[0]],
                        token_frequencies[pair[1]],
                    ),
                    pair_frequencies[pair],
                    pair,
                ),
            )
            merged = self._merged_token(*best_pair)
            splits = {
                segment: self._apply_merge(pieces, best_pair, merged)
                for segment, pieces in splits.items()
            }
            merges.append((best_pair[0], best_pair[1], merged))
            vocabulary.add(merged)

        ordered = list(self.SPECIALS) + sorted(vocabulary.difference(self.SPECIALS))
        self._vocab = {token: index for index, token in enumerate(ordered)}
        self._merges = merges
        self.vocab_size = target_size
        self.min_frequency = minimum
        self._trained = True

    def _encode_segment(self, segment: str) -> list[str]:
        if segment.isspace():
            return [char if char in self._vocab else self.UNK for char in segment]

        tokens: list[str] = []
        position = 0
        while position < len(segment):
            prefix = "" if position == 0 else self.continuation_prefix
            match: str | None = None
            for end in range(len(segment), position, -1):
                candidate = prefix + segment[position:end]
                if candidate in self._vocab:
                    match = candidate
                    position = end
                    break
            if match is None:
                return [self.UNK]
            tokens.append(match)
        return tokens

    def encode(self, text: str) -> list[str]:
        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained.")
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text:
            return []
        tokens: list[str] = []
        for segment in self.pre_tokenize(text):
            tokens.extend(self._encode_segment(segment))
        return tokens

    def decode(self, tokens: list[str]) -> str:
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list of strings.")
        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Every token must be a string.")

        output: list[str] = []
        for token in tokens:
            if token in (self.PAD, self.BOS, self.EOS):
                continue
            if token == self.UNK:
                output.append(self.UNK)
            elif token.startswith(self.continuation_prefix):
                output.append(token[len(self.continuation_prefix):])
            else:
                output.append(token)
        return "".join(output)

    def get_vocab(self) -> dict[str, int]:
        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained.")
        return dict(self._vocab)

    def encode_ids(self, text: str) -> list[int]:
        vocab = self.get_vocab()
        unknown_id = vocab[self.UNK]
        return [vocab.get(token, unknown_id) for token in self.encode(text)]

    def decode_ids(self, ids: list[int]) -> str:
        if not isinstance(ids, list):
            raise TypeError("ids must be a list of integers.")
        if not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in ids
        ):
            raise TypeError("Every ID must be an integer.")
        inverse = {token_id: token for token, token_id in self.get_vocab().items()}
        return self.decode([inverse.get(token_id, self.UNK) for token_id in ids])

    def _get_state(self) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("Cannot save an untrained tokenizer.")
        return {
            "config": {
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "continuation_prefix": self.continuation_prefix,
            },
            "vocab": self._vocab,
            "merges": [list(merge) for merge in self._merges],
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> WordPiece:
        if not isinstance(state, dict):
            raise ValueError("Invalid tokenizer state.")
        config = state.get("config")
        vocab = state.get("vocab")
        merges = state.get("merges", [])
        if not isinstance(config, dict):
            raise ValueError("Missing or invalid tokenizer config.")
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError("Missing or invalid vocabulary.")
        if not isinstance(merges, list):
            raise ValueError("Invalid merge list.")

        instance = cls(**config)
        if any(
            not isinstance(token, str)
            or not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token, token_id in vocab.items()
        ):
            raise ValueError("Invalid vocabulary entries.")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("Vocabulary IDs must be contiguous from zero.")
        if any(special not in vocab for special in cls.SPECIALS):
            raise ValueError("Vocabulary is missing required special tokens.")

        checked_merges: list[tuple[str, str, str]] = []
        for merge in merges:
            if (
                not isinstance(merge, list)
                or len(merge) != 3
                or not all(isinstance(part, str) for part in merge)
            ):
                raise ValueError("Invalid merge entry.")
            checked_merges.append(tuple(merge))

        instance._vocab = dict(vocab)
        instance._merges = checked_merges
        instance._trained = True
        return instance

    def __repr__(self) -> str:
        return f"WordPiece(vocab_size={len(self._vocab)}, merges={len(self._merges)})"
