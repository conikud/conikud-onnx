# conikud-onnx

Hebrew grapheme-to-phoneme (IPA with stress) — ONNX runtime for the Conikud
graph reader. One self-contained `.onnx` file: the wordpiece tokenizer and
the reading graph ride inside it as metadata.

```python
pip install git+https://github.com/conikud/conikud-onnx
```

```python
from conikud_onnx import G2P

g2p = G2P("conikud_int8.onnx")
g2p.phonemize("הלכתי לספר להסתפר")
# halˈaχti lasapˈaʁ lehistapˈeʁ

# speaker/listener gender conditioning (0 unset, 1 male, 2 female)
g2p.phonemize("אני רוצה להגיד לך משהו", speaker=2, listener=2)

# per-word top-k readings with probabilities
g2p.analyze("הוא ספר עד עשר", k=5)
```

`export.py` (dev group: torch/transformers) converts a training checkpoint to
the ONNX file and an int8-quantized variant.
