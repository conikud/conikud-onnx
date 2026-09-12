"""Command line entry point: python -m conikud_onnx [model.onnx] "שלום עולם"."""

from __future__ import annotations

import argparse
import json
import sys

from . import _PROVIDERS, G2P


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="conikud_onnx",
        description="Hebrew grapheme-to-phoneme (IPA with stress) from unvocalized text.")
    parser.add_argument("--model", default=None,
                        help="Local .onnx file (default: download from the Hub)")
    parser.add_argument("text", nargs="?", help="Text to convert (defaults to stdin)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Readings to show per word (>1 prints probabilities)")
    parser.add_argument("--json", action="store_true", help="Emit the full structure")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--providers", nargs="+", default=None,
                        help=f"onnxruntime providers, in order (default: {' '.join(_PROVIDERS)})")
    args = parser.parse_args()

    g2p = G2P(args.model, threads=args.threads, providers=args.providers)

    def process(line: str) -> None:
        line = line.strip()
        if not line:
            return
        if args.json:
            print(json.dumps(g2p.alternatives(line, k=args.top_k), ensure_ascii=False))
        elif args.top_k > 1:
            for word in g2p.alternatives(line, k=args.top_k):
                readings = ", ".join(f"{o['ipa']} ({o['probability']:.2f})" for o in word["options"])
                print(f"{word['word']}: {readings}")
        else:
            print(g2p.phonemize(line))

    if args.text:
        process(args.text)
    else:
        for line in sys.stdin:
            process(line)


if __name__ == "__main__":
    main()
