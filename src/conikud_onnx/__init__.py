"""conikud-onnx — Hebrew grapheme-to-phoneme with the Conikud graph reader.

    from conikud_onnx import G2P
    g2p = G2P("conikud.onnx")
    g2p.phonemize("הלכתי לספר להסתפר")
    g2p.phonemize("אני רוצה להגיד לך משהו", speaker=2, listener=2)  # 0 unset, 1 m, 2 f

Everything (tokenizer, reading graph) is embedded in the .onnx file.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_WORD = re.compile(r"\S+")
_CORE = re.compile(r"[א-ת֑-ׇ׳״'‘’“”\"]+")
_QUOTES = {"׳": "'", "‘": "'", "’": "'",
           "״": '"', "“": '"', "”": '"'}
_HEBREW = {chr(c) for c in range(0x05D0, 0x05EA + 1)} | {"'", '"'}
_ALEF = ord("א")


def _normalize(word: str) -> str:
    word = "".join(c for c in unicodedata.normalize("NFD", word)
                   if unicodedata.category(c) not in ("Mn", "Cf"))
    return "".join(_QUOTES.get(c, c) for c in word)


def _letters(word: str, options: dict) -> list[str]:
    out = []
    for letter, nxt in zip(word, word[1:] + " "):
        if "א" <= letter <= "ת":
            out.append(letter + "'" if nxt == "'" and letter + "'" in options else letter)
    return out


def _letter_id(key: str) -> int:
    return 27 if len(key) > 1 else min(ord(key) - _ALEF, 26)


class G2P:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path)
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.tok = Tokenizer.from_str(meta["tokenizer"])
        self.options = json.loads(meta["options"])
        self.final_het = json.loads(meta["final_het"])
        self.max_letters = int(meta["max_letters"])

    def _word_options(self, keys: list[str]):
        last = len(keys) - 1
        return [self.final_het if i == last and k == "ח" else self.options[k]
                for i, k in enumerate(keys)]

    def _decode(self, opts, cons, vow, stress, k: int = 1):
        """Exact top-k readings (beam over the graph). -> [(ipa, score)] best-first."""
        beams = {False: [(0.0, "")], True: []}
        for l, letter_opts in enumerate(opts):
            new = {False: [], True: []}
            for c, v, plain, stressed in letter_opts:
                base = float(cons[l, c] + vow[l, v])
                for st, entries in beams.items():
                    for score, ipa in entries:
                        new[st].append((score + base + float(stress[l, 0]), ipa + plain))
                        if stressed and not st:
                            new[True].append((score + base + float(stress[l, 1]), ipa + stressed))
            beams = {f: sorted(v_, key=lambda t: -t[0])[:k] for f, v_ in new.items()}
        out = beams[True] or beams[False]
        return [(ipa, score) for score, ipa in out]

    def analyze(self, text: str, ctx: str = "", speaker: int = 0, listener: int = 0,
                k: int = 10):
        """-> token list in input order: {raw, start, end, pre, post, readings};
        readings None for non-Hebrew tokens, else [(ipa, prob)] best-first.
        `ctx` (previous utterances) conditions the model, yields no tokens."""
        import math
        ctx = ctx.strip()
        full = ctx + " \u05c3 " + text if ctx else text
        shift = len(full) - len(text)
        enc = self.tok.encode(full)
        offs = enc.offsets
        tokens, spans, jobs = [], [], []
        for m in _WORD.finditer(text):
            w = m.group()
            entry = {"raw": w, "start": m.start(), "end": m.end(),
                     "pre": "", "post": "", "readings": None}
            c = _CORE.search(w)
            keys = None
            if c:
                heb = _normalize(c.group())
                if heb and all(ch in _HEBREW for ch in heb):
                    keys = _letters(heb, self.options)
                    if not 0 < len(keys) <= self.max_letters:
                        keys = None
            if keys is not None:
                ws, we = m.start() + c.start() + shift, m.start() + c.end() + shift
                ts = [t for t, (a, b) in enumerate(offs) if b > ws and a < we and b > a]
                if ts:
                    entry["pre"], entry["post"] = w[: c.start()], w[c.end():]
                    spans.append((min(ts), max(ts) + 1))
                    jobs.append((len(tokens), keys))
            tokens.append(entry)
        if jobs:
            N = len(jobs)
            lids = np.zeros((N, self.max_letters), dtype=np.int64)
            lens = np.zeros(N, dtype=np.int64)
            for n, (_, keys) in enumerate(jobs):
                lens[n] = len(keys)
                for l, key in enumerate(keys):
                    lids[n, l] = _letter_id(key)
            cons, vow, stress = self.sess.run(
                ["cons", "vow", "stress"],
                {"input_ids": np.array([enc.ids], dtype=np.int64),
                 "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
                 "span_start": np.array([s for s, _ in spans], dtype=np.int64),
                 "span_end": np.array([e for _, e in spans], dtype=np.int64),
                 "letter_ids": lids, "lens": lens,
                 "cond": np.array([[speaker, listener]], dtype=np.int64)})
            for n, (i, keys) in enumerate(jobs):
                opts = self._word_options(keys)
                outs = self._decode(opts, cons[n], vow[n], stress[n], k=k)
                best = max(sc for _, sc in outs)
                ws_ = [math.exp(sc - best) for _, sc in outs]
                z = sum(ws_)
                tokens[i]["readings"] = [(ipa, w_ / z) for (ipa, _), w_ in zip(outs, ws_)]
        return tokens

    def phonemize(self, text: str, ctx: str = "", speaker: int = 0, listener: int = 0) -> str:
        out = []
        for t in self.analyze(text, ctx=ctx, speaker=speaker, listener=listener, k=1):
            if t["readings"]:
                out.append(t["pre"] + t["readings"][0][0] + t["post"])
            else:
                out.append(t["raw"])
        return " ".join(out)
