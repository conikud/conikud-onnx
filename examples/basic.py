"""Phonemize Hebrew text, and inspect the readings behind each word.

The model is fetched from the Hub on first run and cached, so this needs a
network once and nothing afterwards:

    pip install conikud-onnx
    python examples/basic.py

To use a local export instead — no Hub, no network — download it yourself and
pass the path to G2P:

    wget https://huggingface.co/conikud/conikud-onnx/resolve/main/conikud_int8.onnx
"""

from conikud_onnx import G2P

g2p = G2P()

# Hebrew omits vowels, so the same spelling can be several different words.
# ספר is a book, "he counted", or a barber — only the sentence decides.
for sentence in ["קניתי ספר חדש", "הוא ספר את הכסף"]:
    print(f"{sentence}  ->  {g2p.phonemize(sentence)}")

# normalize=True speaks numbers, money, dates and times as words.
print(g2p.phonemize("הכרטיס עלה ₪25 בשעה 14:30", normalize=True))

print()

# alternatives() keeps the competing readings instead of collapsing them,
# each with a probability over the returned options.
for word in g2p.alternatives("הוא ספר את הכסף", k=3):
    readings = "  ".join(f"{o['ipa']} {o['probability']:.0%}" for o in word["options"])
    print(f"{word['word']:>6}  {readings}")
