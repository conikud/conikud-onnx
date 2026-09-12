"""conikud-onnx — Hebrew grapheme-to-phoneme (IPA with stress) from unvocalized text.

    from conikud_onnx import G2P

    g2p = G2P()  # downloads the model from the Hub on first use
    g2p.phonemize("הלכתי לספר להסתפר")       # halˈaχti lasapˈaʁ lehistapˈeʁ
    g2p.alternatives("הוא ספר עד עשר", k=5)  # per-word readings with probabilities

The chunk vocabulary, letter rules and tokenizer all ride inside the .onnx
file's metadata, so a single file is the whole runtime.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import regex as re
from heb_tts_normalizer import normalize as _spoken_form
from tokenizers import Tokenizer

from .hub import resolve

__all__ = ["G2P"]

_BIDI = re.compile(r"[\p{Bidi_Control}﻿]")
_HEBREW_MARKS = re.compile(r"[\p{M}&&\p{scx=Hebrew}]", re.V1)
_WORD = re.compile(r"\S+")
_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def _normalize(text: str) -> str:
    """Strip niqqud, bidi controls and repeated whitespace."""
    text = _HEBREW_MARKS.sub("", unicodedata.normalize("NFD", _BIDI.sub("", text)))
    return " ".join(unicodedata.normalize("NFC", text).split())


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class _Vocabulary:
    """Chunk and letter rules recovered from the .onnx file's metadata_props."""

    def __init__(self, props: dict[str, str]):
        if props.get("g2pw.format") != "1":
            raise ValueError("Not a g2pw model file: no g2pw metadata found")
        self.chunks = tuple(json.loads(props["g2pw.chunks"]))
        self.stressed_chunks = tuple(json.loads(props["g2pw.stressed_chunks"]))
        self.letter_sounds = {k: tuple(v) for k, v in json.loads(props["g2pw.letter_sounds"]).items()}
        self.letter_chunks = {k: tuple(v) for k, v in json.loads(props["g2pw.letter_chunks"]).items()}
        self.final_chunks = {k: tuple(v) for k, v in json.loads(props["g2pw.final_chunks"]).items()}
        self.markers = props["g2pw.markers"]
        self.max_chars = int(props["g2pw.max_chars"])
        self.max_tokens = int(props["g2pw.max_tokens"])
        ids = {chunk: i for i, chunk in enumerate(self.chunks)}
        self.has_vowel = tuple(any(v in chunk for v in "aeiou") for chunk in self.chunks)
        # Per-letter chunk ids, split by whether the letter ends its word.
        self.allowed = {c: tuple(ids[k] for k in self.letter_chunks[c]) for c in self.letter_chunks}
        self.allowed_final = {c: tuple(ids[k] for k in self.letter_chunks[c] + self.final_chunks.get(c, ()))
                              for c in self.letter_chunks}

    def char_id(self, char: str) -> int:
        return ord(char) - ord("א") + 3 if char in self.letter_sounds else (2 if char.isspace() else 1)

    def word_final_positions(self, text: str) -> list[bool]:
        result = [False] * len(text)
        for match in _WORD.finditer(text):
            letters = [i for i in range(*match.span()) if text[i] in self.letter_sounds]
            if letters:
                result[letters[-1]] = True
        return result

    def render(self, text: str, start: int, end: int,
               chunks: dict[int, int], stressed: set[int]) -> str:
        """Spell one word, dropping orthographic marks attached to Hebrew letters."""
        out = []
        for i in range(start, end):
            char = text[i]
            if char not in self.letter_sounds:
                attached = any(c in self.letter_sounds for c in text[max(0, i - 1):i + 2])
                out.append("" if char in self.markers and attached else char)
            else:
                cid = chunks[i]
                out.append(self.stressed_chunks[cid] if i in stressed else self.chunks[cid])
        return "".join(out)


class G2P:
    """Hebrew grapheme-to-phoneme over a self-contained .onnx file.

    `model_path` is optional: with no argument the int8 model is downloaded
    from the Hub (`conikud/conikud-onnx`) and cached, so later calls are
    offline. Pass a path to use your own export instead.

    `providers` defaults to CUDA when available, else CPU. Note that int8
    models run best on CPU: dynamic quantization leaves integer matmuls
    without native CUDA kernels, so they fall back across the device boundary.
    """

    def __init__(self, model_path: str | Path | None = None, threads: int | None = None,
                 providers: list[str] | None = None):
        model_path = resolve(model_path)
        props = {e.key: e.value for e in
                 onnx.load(model_path, load_external_data=False).metadata_props}
        self.vocab = _Vocabulary(props)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        available = ort.get_available_providers()
        chosen = [p for p in (providers or _PROVIDERS) if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, options, providers=chosen)
        self.tokenizer = self._load_tokenizer(props)

    @staticmethod
    def _load_tokenizer(props: dict[str, str]) -> Tokenizer:
        files = json.loads(props["g2pw.tokenizer"])
        if "tokenizer.json" in files:
            return Tokenizer.from_str(files["tokenizer.json"])
        # Older exports carry only the slow-tokenizer files; convert once.
        with TemporaryDirectory() as temp:
            for name, content in files.items():
                Path(temp, name).write_text(content, encoding="utf-8")
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(temp).backend_tokenizer

    def _run(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Character-aligned chunk and stress log-probabilities for one sentence."""
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        if len(encoded.ids) > self.vocab.max_tokens:
            raise ValueError(f"Text is too long; maximum {self.vocab.max_tokens} tokens")
        if len(text) > self.vocab.max_chars:
            raise ValueError(f"Text is too long; maximum {self.vocab.max_chars} characters")

        char_to_token = [-1] * len(text)
        char_positions = [0] * len(text)
        for token, (start, end) in enumerate(encoded.offsets):
            for pos in range(start, end):
                char_to_token[pos] = token
                char_positions[pos] = pos - start

        emissions, stress, _ = self.session.run(["emissions", "stress", "prefix"], {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "char_ids": np.array([[self.vocab.char_id(c) for c in text]], dtype=np.int64),
            "char_to_token": np.array([char_to_token], dtype=np.int64),
            "char_positions": np.array([char_positions], dtype=np.int64),
            "word_final": np.array([self.vocab.word_final_positions(text)], dtype=np.int64),
        })
        return _log_softmax(emissions[0]), _log_softmax(stress[0])

    def _readings(self, text: str, start: int, end: int, emissions: np.ndarray,
                  stress: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Exact top-k readings of one word: a beam over each letter's legal chunks.

        Beams are split by whether a stress has been placed, so every surviving
        reading carries exactly one stress — Hebrew words always bear one.
        """
        letters = [i for i in range(start, end) if text[i] in self.vocab.letter_sounds]
        if not letters:
            return []
        last = letters[-1]
        beams: dict[bool, list[tuple[float, tuple[tuple[int, int], ...]]]] = {False: [(0.0, ())], True: []}
        for i in letters:
            allowed = (self.vocab.allowed_final if i == last else self.vocab.allowed)[text[i]]
            nxt: dict[bool, list[tuple[float, tuple[tuple[int, int], ...]]]] = {False: [], True: []}
            for cid in allowed:
                base = float(emissions[i, cid])
                plain, marked = base + float(stress[i, 0]), base + float(stress[i, 1])
                voweled = self.vocab.has_vowel[cid]
                for placed, entries in beams.items():
                    for score, path in entries:
                        nxt[placed].append((score + plain, path + ((cid, 0),)))
                        if voweled and not placed:
                            nxt[True].append((score + marked, path + ((cid, 1),)))
            beams = {placed: sorted(v, key=lambda t: -t[0])[:k] for placed, v in nxt.items()}
        # A word with no vowel at all never reaches the stressed beam.
        best = beams[True] or beams[False]

        readings = []
        for score, path in best:
            chunks = {i: cid for i, (cid, _) in zip(letters, path)}
            stressed = {i for i, (_, s) in zip(letters, path) if s}
            readings.append((self.vocab.render(text, start, end, chunks, stressed), score))
        return readings

    def alternatives(self, text: str, k: int = 5, normalize: bool = False) -> list[dict]:
        """Per-word readings in input order, best first.

        -> [{word, start, end, options: [{ipa, score, probability}]}]. `options`
        is empty for tokens with no Hebrew letters. `score` is a summed
        log-probability; `probability` is a softmax over the returned readings,
        so it sums to 1 per word.

        `normalize=True` first rewrites numbers, money, dates and units into the
        words a person would say — otherwise digits reach the
        model as characters it has no reading for. It rewrites the
        text, so `word` and the offsets then describe the spoken form rather
        than what was passed in.
        """
        if normalize:
            text = _spoken_form(text)
        text = _normalize(text)
        if not text:
            return []
        emissions, stress = self._run(text)

        results = []
        for match in _WORD.finditer(text):
            start, end = match.span()
            readings = self._readings(text, start, end, emissions, stress, k)
            options = []
            if readings:
                top = max(score for _, score in readings)
                weights = [float(np.exp(score - top)) for _, score in readings]
                total = sum(weights)
                options = [{"ipa": ipa, "score": score, "probability": w / total}
                           for (ipa, score), w in zip(readings, weights)]
            results.append({"word": match.group(), "start": start, "end": end, "options": options})
        return results

    def phonemize(self, text: str, normalize: bool = False) -> str:
        """Best reading of every word, joined by spaces.

        `normalize=True` speaks numbers, money, dates and units as words first;
        see :meth:`alternatives`.
        """
        return " ".join(word["options"][0]["ipa"] if word["options"] else word["word"]
                        for word in self.alternatives(text, k=1, normalize=normalize))
