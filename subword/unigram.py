from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, ClassVar

from langtoken.base import Tokenizer, TokenizerMetadata


class Unigram(Tokenizer):
    """
    Unigram Language Model tokenizer.

    Uses candidate generation, Expectation-Maximization and
    vocabulary pruning to learn a probabilistic subword vocabulary.

    Individual characters are protected as fallback tokens.
    Whitespace is represented independently, so subword candidates
    cannot cross word boundaries.
    """

    UNK: ClassVar[str] = "<UNK>"

    META: ClassVar[TokenizerMetadata] = TokenizerMetadata(
        name="Unigram",
        family="subword",
        version="1.1.0",
        description=(
            "Unigram language model tokenizer using EM, "
            "probability constraints and vocabulary pruning."
        ),
    )

    def __init__(
        self,
        vocab_size: int = 8000,
        max_piece_length: int = 16,
        shrinking_factor: float = 0.8,
        em_iterations: int = 20,
        candidate_multiplier: int = 4,
        min_frequency: int = 2,
        em_tolerance: float = 1e-5,
        prune_eval_limit: int = 256,
        smoothing: float = 0.1,
        char_probability_floor: float = 1e-4,
        max_token_probability: float = 0.2,
    ) -> None:

        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")

        if max_piece_length < 1:
            raise ValueError("max_piece_length must be positive.")

        if not 0 < shrinking_factor < 1:
            raise ValueError("shrinking_factor must be between 0 and 1.")

        if em_iterations < 1:
            raise ValueError("em_iterations must be positive.")

        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive.")

        if min_frequency < 1:
            raise ValueError("min_frequency must be positive.")

        if em_tolerance < 0:
            raise ValueError("em_tolerance cannot be negative.")

        if prune_eval_limit < 1:
            raise ValueError("prune_eval_limit must be positive.")

        if smoothing <= 0 or not math.isfinite(smoothing):
            raise ValueError("smoothing must be finite and positive.")

        if (
            char_probability_floor <= 0
            or not math.isfinite(char_probability_floor)
        ):
            raise ValueError(
                "char_probability_floor must be finite and positive."
            )

        if (
            not 0 < max_token_probability <= 1
            or not math.isfinite(max_token_probability)
        ):
            raise ValueError(
                "max_token_probability must be in (0, 1]."
            )

        self.vocab_size = vocab_size
        self.max_piece_length = max_piece_length
        self.shrinking_factor = shrinking_factor
        self.em_iterations = em_iterations
        self.candidate_multiplier = candidate_multiplier
        self.min_frequency = min_frequency
        self.em_tolerance = em_tolerance
        self.prune_eval_limit = prune_eval_limit

        self.smoothing = smoothing
        self.char_probability_floor = char_probability_floor
        self.max_token_probability = max_token_probability

        self._log_probs: dict[str, float] = {}
        self._index: dict[str, list[str]] = {}
        self._trained = False
        self._last_expected_counts: dict[str, float] = {}

    # ---------------------------------------------------------
    # Numerical utilities
    # ---------------------------------------------------------

    @staticmethod
    def _logsumexp(values: list[float]) -> float:
        """Numerically stable log-sum-exp."""

        if not values:
            return -math.inf

        maximum = max(values)

        if maximum == -math.inf:
            return -math.inf

        return maximum + math.log(
            sum(math.exp(value - maximum) for value in values)
        )

    @classmethod
    def _normalize_scores(
        cls,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Normalize positive weights into log probabilities."""

        if not scores:
            raise ValueError("Cannot normalize an empty vocabulary.")

        if any(
            not math.isfinite(score) or score <= 0
            for score in scores.values()
        ):
            raise ValueError(
                "Token weights must be finite and positive."
            )

        log_scores = {
            token: math.log(score)
            for token, score in scores.items()
        }

        normalizer = cls._logsumexp(list(log_scores.values()))

        return {
            token: score - normalizer
            for token, score in log_scores.items()
        }

    def _constrain_probabilities(
        self,
        log_probs: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply lower and upper probability bounds.

        Individual characters receive at least
        char_probability_floor.

        Every token is limited to max_token_probability.

        A bounded redistribution algorithm ensures the final
        probabilities still sum to one.
        """

        if not log_probs:
            raise ValueError("Cannot constrain an empty vocabulary.")

        tokens = list(log_probs)

        char_tokens = [
            token
            for token in tokens
            if len(token) == 1 and token != self.UNK
        ]

        floor = self.char_probability_floor
        ceiling = self.max_token_probability

        if len(char_tokens) * floor >= 1:
            raise ValueError(
                "char_probability_floor is too large for "
                "the number of character tokens."
            )

        if len(tokens) * ceiling < 1:
            raise ValueError(
                "max_token_probability is too small for "
                "the vocabulary size."
            )

        weights = {
            token: math.exp(log_probs[token])
            for token in tokens
        }

        total = sum(weights.values())

        if not math.isfinite(total) or total <= 0:
            raise ValueError("Invalid probability distribution.")

        weights = {
            token: weight / total
            for token, weight in weights.items()
        }

        probabilities: dict[str, float] = {}
        fixed: set[str] = set()

        # Tokens violating their lower or upper bounds are fixed.
        for _ in range(len(tokens) + 1):

            remaining_mass = 1.0 - sum(probabilities.values())

            free_tokens = [
                token
                for token in tokens
                if token not in fixed
            ]

            if not free_tokens:
                break

            free_weight = sum(
                weights[token]
                for token in free_tokens
            )

            if free_weight <= 0:
                raise ValueError(
                    "Unable to redistribute probability mass."
                )

            proposed = {
                token: remaining_mass
                * weights[token]
                / free_weight
                for token in free_tokens
            }

            violations = {}

            for token, probability in proposed.items():

                if token in char_tokens and probability < floor:
                    violations[token] = floor

                elif probability > ceiling:
                    violations[token] = ceiling

            if not violations:
                probabilities.update(proposed)
                break

            for token, probability in violations.items():
                probabilities[token] = probability
                fixed.add(token)

        else:
            raise RuntimeError(
                "Probability constraint redistribution failed."
            )

        total = sum(probabilities.values())

        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(
                f"Probability distribution does not sum to one: {total}"
            )

        return {
            token: math.log(probability)
            for token, probability in probabilities.items()
        }

    # ---------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------

    @staticmethod
    def _split_corpus(corpus: str | list[str]) -> list[str]:
        """Normalize the input corpus into documents."""

        if isinstance(corpus, str):
            documents = [corpus]

        elif isinstance(corpus, list) and all(
            isinstance(document, str) for document in corpus
        ):
            documents = corpus

        else:
            raise TypeError(
                "corpus must be a string or a list of strings."
            )

        if not documents or not any(documents):
            raise ValueError("Cannot train on an empty corpus.")

        return documents

    def _generate_candidates(
        self,
        corpus: list[str],
    ) -> dict[str, float]:
        """
        Generate candidate tokens.

        Single characters are mandatory.
        Multi-character candidates are generated within words.
        Whitespace is retained independently.
        """

        counts: Counter[str] = Counter()
        mandatory: set[str] = set()

        for document in corpus:

            for segment in re.findall(r"\s+|\S+", document):

                mandatory.update(segment)

                if segment.isspace():
                    counts.update(segment)
                    continue

                length = len(segment)

                for start in range(length):

                    end_limit = min(
                        length,
                        start + self.max_piece_length,
                    )

                    for end in range(start + 1, end_limit + 1):
                        counts[segment[start:end]] += 1

        mandatory.add(self.UNK)

        if len(mandatory) > self.vocab_size:
            raise ValueError(
                f"Mandatory vocabulary size ({len(mandatory)}) "
                f"exceeds requested vocab_size ({self.vocab_size})."
            )

        candidates = [
            token
            for token, count in counts.items()
            if len(token) > 1
            and count >= self.min_frequency
        ]

        candidates.sort(
            key=lambda token: (
                -counts[token],
                -len(token),
                token,
            )
        )

        candidate_limit = (
            self.vocab_size * self.candidate_multiplier
        )

        remaining_slots = max(
            0,
            candidate_limit - len(mandatory),
        )

        selected = candidates[:remaining_slots]

        scores = {
            token: float(counts[token])
            for token in mandatory
            if token != self.UNK
        }

        for token in selected:
            scores[token] = float(counts[token])

        # A small initial UNK weight keeps it available without
        # making it competitive with ordinary vocabulary tokens.
        scores[self.UNK] = self.smoothing

        return scores

    # ---------------------------------------------------------
    # Vocabulary indexing
    # ---------------------------------------------------------

    def _build_index(self) -> None:
        """Group tokens by their first character."""

        index: dict[str, list[str]] = defaultdict(list)

        for token in self._log_probs:

            if token == self.UNK:
                continue

            index[token[0]].append(token)

        self._index = {
            character: sorted(
                tokens,
                key=lambda token: (-len(token), token),
            )
            for character, tokens in index.items()
        }

    def _matches(self, text: str, position: int):
        """
        Yield all known matches at a position.

        UNK is only available if no known token matches the
        current character position.
        """

        character = text[position]

        matches = []

        for token in self._index.get(character, []):

            if text.startswith(token, position):
                matches.append(token)

        if matches:

            for token in matches:
                yield (
                    position + len(token),
                    token,
                    self._log_probs[token],
                )

        else:

            yield (
                position + 1,
                self.UNK,
                self._log_probs[self.UNK],
            )

    # ---------------------------------------------------------
    # Forward and backward algorithms
    # ---------------------------------------------------------

    def _forward(self, text: str) -> list[float]:
        """Calculate forward log probabilities."""

        length = len(text)

        forward = [-math.inf] * (length + 1)
        forward[0] = 0.0

        for position in range(length):

            if forward[position] == -math.inf:
                continue

            for end, _, score in self._matches(text, position):

                forward[end] = self._logsumexp([
                    forward[end],
                    forward[position] + score,
                ])

        return forward

    def _backward(self, text: str) -> list[float]:
        """Calculate backward log probabilities."""

        length = len(text)

        backward = [-math.inf] * (length + 1)
        backward[length] = 0.0

        for position in range(length - 1, -1, -1):

            candidates = []

            for end, _, score in self._matches(text, position):
                candidates.append(
                    score + backward[end]
                )

            backward[position] = self._logsumexp(candidates)

        return backward

    def _forward_backward(
        self,
        text: str,
        collect_counts: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Calculate text likelihood and optional expected counts."""

        if not text:
            return 0.0, {}

        forward = self._forward(text)
        backward = self._backward(text)

        log_likelihood = forward[len(text)]

        if log_likelihood == -math.inf:
            raise RuntimeError(
                "Unable to segment text with the current vocabulary."
            )

        if not collect_counts:
            return log_likelihood, {}

        expected_counts: dict[str, float] = defaultdict(float)

        for position in range(len(text)):

            for end, token, score in self._matches(
                text,
                position,
            ):

                log_probability = (
                    forward[position]
                    + score
                    + backward[end]
                    - log_likelihood
                )

                expected_counts[token] += math.exp(
                    log_probability
                )

        return log_likelihood, dict(expected_counts)

    # ---------------------------------------------------------
    # Expectation-Maximization
    # ---------------------------------------------------------

    def _run_em(self, corpus: list[str]) -> None:
        """
        Estimate probabilities using EM.

        Additive smoothing prevents zero expected counts from
        eliminating tokens completely.

        Probability constraints preserve character fallback
        and prevent individual tokens from dominating.
        """

        for _ in range(self.em_iterations):

            expected_counts: dict[str, float] = defaultdict(float)

            for document in corpus:

                _, counts = self._forward_backward(
                    document,
                    collect_counts=True,
                )

                for token, count in counts.items():
                    expected_counts[token] += count

            self._last_expected_counts = dict(expected_counts)

            scores = {
                token: (
                    expected_counts.get(token, 0.0)
                    + self.smoothing
                )
                for token in self._log_probs
            }

            new_log_probs = self._normalize_scores(scores)

            new_log_probs = self._constrain_probabilities(
                new_log_probs
            )

            maximum_change = max(
                abs(
                    new_log_probs[token]
                    - self._log_probs[token]
                )
                for token in self._log_probs
            )

            self._log_probs = new_log_probs
            self._build_index()

            if maximum_change < self.em_tolerance:
                break

    # ---------------------------------------------------------
    # Corpus likelihood and pruning
    # ---------------------------------------------------------

    def _corpus_loss(self, corpus: list[str]) -> float:
        """Calculate negative corpus log likelihood."""

        total_likelihood = 0.0

        for document in corpus:
            likelihood, _ = self._forward_backward(document)
            total_likelihood += likelihood

        return -total_likelihood

    def _prune(self, corpus: list[str]) -> None:
        """
        Remove low-impact multi-character candidates.

        Single-character tokens and UNK are never pruned.

        Candidates are shortlisted by expected usage, then
        evaluated by their impact on corpus likelihood.
        """

        removable = [
            token
            for token in self._log_probs
            if len(token) > 1 and token != self.UNK
        ]

        excess = len(self._log_probs) - self.vocab_size

        if excess <= 0:
            return

        if not removable:
            raise RuntimeError(
                "Unable to reach the requested vocabulary size."
            )

        remove_count = min(
            excess,
            max(
                1,
                math.ceil(
                    len(removable) * (1 - self.shrinking_factor)
                ),
            ),
        )

        removable.sort(
            key=lambda token: (
                self._last_expected_counts.get(token, 0.0),
                token,
            )
        )

        shortlist_size = min(
            len(removable),
            max(
                self.prune_eval_limit,
                remove_count * 3,
            ),
        )

        shortlist = removable[:shortlist_size]

        baseline_loss = self._corpus_loss(corpus)

        impacts: list[tuple[float, str]] = []

        original_log_probs = self._log_probs

        for token in shortlist:

            remaining = {
                piece: probability
                for piece, probability in original_log_probs.items()
                if piece != token
            }

            normalizer = self._logsumexp(
                list(remaining.values())
            )

            candidate_log_probs = {
                piece: probability - normalizer
                for piece, probability in remaining.items()
            }

            candidate_log_probs = self._constrain_probabilities(
                candidate_log_probs
            )

            self._log_probs = candidate_log_probs
            self._build_index()

            try:
                loss_after_removal = self._corpus_loss(corpus)
                impact = loss_after_removal - baseline_loss

            finally:
                self._log_probs = original_log_probs
                self._build_index()

            impacts.append((impact, token))

        impacts.sort(
            key=lambda item: (item[0], item[1])
        )

        tokens_to_remove = {
            token
            for _, token in impacts[:remove_count]
        }

        self._log_probs = {
            token: probability
            for token, probability in original_log_probs.items()
            if token not in tokens_to_remove
        }

        # Reapply constraints after pruning.
        self._log_probs = self._constrain_probabilities(
            self._log_probs
        )

        self._build_index()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(
        self,
        corpus: str | list[str],
        **kwargs: Any,
    ) -> None:
        """
        Train the Unigram tokenizer.

        Constructor configuration can be overridden through kwargs.
        """

        allowed_options = {
            "vocab_size",
            "max_piece_length",
            "shrinking_factor",
            "em_iterations",
            "candidate_multiplier",
            "min_frequency",
            "em_tolerance",
            "prune_eval_limit",
            "smoothing",
            "char_probability_floor",
            "max_token_probability",
        }

        unknown_options = set(kwargs) - allowed_options

        if unknown_options:
            raise TypeError(
                f"Unexpected training arguments: "
                f"{sorted(unknown_options)}"
            )

        if kwargs:
            config = {
                "vocab_size": self.vocab_size,
                "max_piece_length": self.max_piece_length,
                "shrinking_factor": self.shrinking_factor,
                "em_iterations": self.em_iterations,
                "candidate_multiplier": self.candidate_multiplier,
                "min_frequency": self.min_frequency,
                "em_tolerance": self.em_tolerance,
                "prune_eval_limit": self.prune_eval_limit,
                "smoothing": self.smoothing,
                "char_probability_floor": self.char_probability_floor,
                "max_token_probability": self.max_token_probability,
            }

            config.update(kwargs)

            # Reuse constructor validation.
            validated = type(self)(**config)
            self.__dict__.update(validated.__dict__)

        documents = self._split_corpus(corpus)

        scores = self._generate_candidates(documents)

        if len(scores) < self.vocab_size:
            self.vocab_size = len(scores)

        self._log_probs = self._normalize_scores(scores)

        self._log_probs = self._constrain_probabilities(
            self._log_probs
        )

        self._build_index()

        while len(self._log_probs) > self.vocab_size:

            self._run_em(documents)

            previous_size = len(self._log_probs)

            self._prune(documents)

            if len(self._log_probs) >= previous_size:
                raise RuntimeError(
                    "Vocabulary pruning failed to remove candidates."
                )

        self._run_em(documents)

        self._trained = True

    # ---------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[str]:
        """Find the highest-probability tokenization using Viterbi."""

        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained.")

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text:
            return []

        length = len(text)

        best_scores = [-math.inf] * (length + 1)
        best_scores[0] = 0.0

        backpointers: list[
            tuple[int, str] | None
        ] = [None] * (length + 1)

        for position in range(length):

            if best_scores[position] == -math.inf:
                continue

            for end, token, score in self._matches(
                text,
                position,
            ):

                candidate_score = (
                    best_scores[position] + score
                )

                if candidate_score > best_scores[end]:

                    best_scores[end] = candidate_score
                    backpointers[end] = (
                        position,
                        token,
                    )

        tokens = []
        position = length

        while position > 0:

            pointer = backpointers[position]

            if pointer is None:
                raise RuntimeError(
                    "Unable to reconstruct tokenization."
                )

            previous_position, token = pointer

            tokens.append(token)
            position = previous_position

        tokens.reverse()

        return tokens

    # ---------------------------------------------------------
    # Decoding
    # ---------------------------------------------------------

    def decode(self, tokens: list[str]) -> str:
        """Reconstruct text by concatenating tokens."""

        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list of strings.")

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Every token must be a string.")

        return "".join(tokens)

    # ---------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary with stable token IDs."""

        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained.")

        tokens = sorted(
            (
                token
                for token in self._log_probs
                if token != self.UNK
            ),
            key=lambda token: (
                -self._log_probs[token],
                token,
            ),
        )

        return {
            self.UNK: 0,
            **{
                token: index + 1
                for index, token in enumerate(tokens)
            },
        }

    # ---------------------------------------------------------
    # Probability inspection
    # ---------------------------------------------------------

    def get_token_probabilities(self) -> dict[str, float]:
        """Return token probabilities in ordinary probability space."""

        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained.")

        return {
            token: math.exp(log_probability)
            for token, log_probability in self._log_probs.items()
        }

    def get_log_probabilities(self) -> dict[str, float]:
        """Return learned log probabilities."""

        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained.")

        return self._log_probs.copy()

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def _get_state(self) -> dict[str, Any]:
        """Return the state required for serialization."""

        if not self._trained:
            raise RuntimeError(
                "Cannot save an untrained tokenizer."
            )

        return {
            "config": {
                "vocab_size": self.vocab_size,
                "max_piece_length": self.max_piece_length,
                "shrinking_factor": self.shrinking_factor,
                "em_iterations": self.em_iterations,
                "candidate_multiplier": self.candidate_multiplier,
                "min_frequency": self.min_frequency,
                "em_tolerance": self.em_tolerance,
                "prune_eval_limit": self.prune_eval_limit,
                "smoothing": self.smoothing,
                "char_probability_floor": self.char_probability_floor,
                "max_token_probability": self.max_token_probability,
            },
            "log_probs": self._log_probs,
        }

    @classmethod
    def _from_state(
        cls,
        state: dict[str, Any],
    ) -> Unigram:

        if not isinstance(state, dict):
            raise ValueError("Invalid tokenizer state.")

        config = state.get("config")
        log_probs = state.get("log_probs")

        if not isinstance(config, dict):
            raise ValueError("Missing or invalid tokenizer config.")

        if not isinstance(log_probs, dict) or not log_probs:
            raise ValueError("Missing or invalid token probabilities.")

        if cls.UNK not in log_probs:
            raise ValueError("Tokenizer state is missing the UNK token.")

        for token, probability in log_probs.items():

            if (
                not isinstance(token, str)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
            ):
                raise ValueError("Invalid token probability.")

        instance = cls(**config)

        instance._log_probs = {
            token: float(probability)
            for token, probability in log_probs.items()
        }

        # Validate that loaded probabilities are normalized and
        # satisfy the configured constraints.
        total = sum(
            math.exp(probability)
            for probability in instance._log_probs.values()
        )

        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Loaded token probabilities are not normalized."
            )

        constrained = instance._constrain_probabilities(
            instance._log_probs
        )

        if any(
            not math.isclose(
                constrained[token],
                instance._log_probs[token],
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            for token in instance._log_probs
        ):
            raise ValueError(
                "Loaded probabilities violate configured constraints."
            )

        instance._build_index()
        instance._trained = True

        return instance
