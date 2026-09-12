# conikud-onnx

Hebrew grapheme-to-phoneme — IPA with stress, from plain unvocalized text.
One self-contained `.onnx` file: the wordpiece tokenizer, the chunk vocabulary
and the per-letter phonetic rules all ride inside it as metadata. No torch, no
checkpoint directory, no lexicon.

```sh
pip install git+https://github.com/conikud/conikud-onnx
```

## Usage

```python
from conikud_onnx import G2P

g2p = G2P()  # downloads conikud/conikud-onnx on first use, then cached

g2p.phonemize("הלכתי לספר להסתפר")
# halˈaχti lasapˈaʁ lehistapˈeʁ
```

Hebrew is written without vowels, so spelling alone rarely fixes a reading.
`ספר` is *sˈefeʁ* (a book), *safˈaʁ* (he counted) or *sapˈaʁ* (a barber) —
only the sentence decides:

```python
g2p.phonemize("קניתי ספר חדש")     # kanˈiti sˈefeʁ χadˈaʃ
g2p.phonemize("הוא ספר את הכסף")   # hˈu safˈaʁ ʔˈet hakˈesef
```

### Alternatives

`alternatives()` returns every word's competing readings with probabilities, rather
than collapsing them to a single answer:

```python
g2p.alternatives("הוא ספר את הכסף", k=3)
```

```python
[{"word": "הוא", "start": 0, "end": 3, "options": [
     {"ipa": "hˈu", "score": -0.01, "probability": 1.00}, ...]},
 {"word": "ספר", "start": 4, "end": 7, "options": [
     {"ipa": "safˈaʁ", "score": -0.93, "probability": 0.62},
     {"ipa": "safˈeʁ", "score": -1.85, "probability": 0.25},
     {"ipa": "sapˈaʁ", "score": -2.53, "probability": 0.13}]},
 ...]
```

Readings are best-first. `score` is a summed log-probability; `probability` is
a softmax over the returned readings, so it sums to 1 per word. `options` is
empty for tokens with no Hebrew letters, and `start`/`end` index the
normalized text.

Each reading is an exact top-k beam over the chunks each letter is allowed to
emit, carrying exactly one stress per word — so every variant is a
well-formed pronunciation of that spelling, never an arbitrary logit sample.

## Hardware

The session prefers CUDA and falls back to CPU:

```python
G2P(providers=["CPUExecutionProvider"], threads=4)
```

int8 weights are the right default on CPU. On GPU, prefer the fp32 file:
dynamic quantization leaves the integer matmuls without native CUDA kernels,
so they fall back across the device boundary and lose more to memory copies
than quantization saves.

## Model

The int8 weights live at
[`conikud/conikud-onnx`](https://huggingface.co/conikud/conikud-onnx) and are
fetched automatically. To pin your own export, pass a path:

```python
G2P("conikud_int8.onnx")
```

See [`examples/basic.py`](examples/basic.py) for a runnable version of the above.
