
# LangToken

A modular Python library for text tokenization.

LangToken provides implementations of several tokenization
algorithms, organized into three main families:

- Subword tokenizers
- Character tokenizers
- Word tokenizers

## Installation

```bash
pip install langtoken-toolkit
```

## Features

### Subword Tokenizers
- BPE
- Byte-Level BPE
- WordPiece
- Unigram
- SentencePiece

### Character Tokenizers
- Character Tokenizer
- Character Shingle

### Word Tokenizers
- Word Tokenizer
- Whitespace Word Tokenizer
- Standard Word Tokenizer

## Quick Start

```python
from langtoken import BPE

corpus = [
    "The quick brown fox",
    "The fox jumps over the dog",
]

tokenizer = BPE()

tokenizer.train(
    corpus,
    vocab_size=100,
)

text = "The quick brown fox"

tokens = tokenizer.encode(text)
ids = tokenizer.encode_ids(text)

print("Tokens:", tokens)
print("IDs:", ids)

print("Decoded:", tokenizer.decode(tokens))
print("Decoded IDs:", tokenizer.decode_ids(ids))
```

## Saving and Loading

```python
tokenizer.save("tokenizer.json")

loaded = BPE.load("tokenizer.json")

print(loaded.encode("The quick brown fox"))
```

## Project Structure

```text
langtoken/
    base.py
    subword/
    char/
    word/
```

## License

MIT License
