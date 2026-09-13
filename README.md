# conikud-onnx

Hebrew grapheme-to-phoneme — IPA with stress, from plain unvocalized text.
One self-contained `.onnx` file: tokenizer, chunk vocabulary and letter rules
all ride inside it as metadata.

```sh
pip install git+https://github.com/conikud/conikud-onnx
```

```python
from conikud_onnx import G2P

g2p = G2P()  # fetches conikud/conikud-onnx on first use, then cached

g2p.phonemize("הלכתי לספר להסתפר")
# halˈaχti lasapˈaʁ lehistapˈeʁ
```

Hebrew omits vowels, so spelling alone rarely fixes a reading. `ספר` is
*sˈefeʁ* (a book), *safˈaʁ* (he counted) or *sapˈaʁ* (a barber) — only the
sentence decides:

```python
g2p.phonemize("קניתי ספר חדש")     # kanˈiti sˈefeʁ χadˈaʃ
g2p.phonemize("הוא ספר את הכסף")   # hˈu safˈaʁ ʔˈet hakˈesef
```

## Gender

Hebrew inflects for the gender of **both** participants, and the spelling is often
identical — `לך` is *leχˈa* to a man and *lˈaχ* to a woman. No lexicon fixes this:
the reading depends on metadata that is not in the text.

```python
g2p.phonemize("זה שלך", listener="m")   # zˈe ʃelχˈa
g2p.phonemize("זה שלך", listener="f")   # zˈe ʃelˈaχ
g2p.phonemize("אני רוצה", speaker="f")  # ʔanˈi ʁotsˈa
```

`speaker` is who is talking, `listener` who is addressed; both take `"m"`, `"f"`
or `None`. Omitted means unknown — the model reads that as *no signal*, not as a
third value, and falls back to inferring gender from context.

Numbers, money, dates and times are spoken as words with `normalize=True`:

```python
g2p.phonemize("הכרטיס עלה ₪25 בשעה 14:30", normalize=True)
# hakaʁtˈis ʔalˈa ʔesʁˈim veχamiʃˈa ʃkalˈim beʃaʔˈa ʃtˈajim vaχˈetsi ʔaχˈaʁ hatsohoʁˈajim
```

It rewrites the text, so `alternatives()` then reports the spoken words and
their offsets rather than what you passed in.

## Alternatives

`alternatives()` keeps each word's competing readings instead of collapsing
them to one:

```python
g2p.alternatives("הוא ספר את הכסף", k=3)
# ספר -> safˈaʁ 0.62, safˈeʁ 0.25, sapˈaʁ 0.13
```

Each word yields `{word, start, end, options}`, with `options` best-first as
`{ipa, score, probability}` — `score` a summed log-probability, `probability`
a softmax over the returned readings. Tokens without Hebrew letters get no
options.

Readings come from an exact top-k beam over the chunks each letter may emit,
carrying exactly one stress per word, so every variant is a well-formed
pronunciation of that spelling.

## Options

```python
G2P("conikud_int8.onnx")                                # a local export
G2P(providers=["CPUExecutionProvider"], threads=4)      # default: CUDA, else CPU
```

int8 is the right default on CPU. On GPU prefer an fp32 export: quantized
matmuls have no native CUDA kernels and fall back across the device boundary,
losing more to memory copies than quantization saves.

See [`examples/basic.py`](examples/basic.py).
