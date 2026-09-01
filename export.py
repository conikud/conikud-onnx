"""Export a GraphReader checkpoint to one self-contained ONNX file.

The file embeds everything the runtime needs as metadata: the wordpiece
tokenizer (JSON) and the reading graph's option tables. Run from this folder
with the dev group (torch/transformers/safetensors/onnx):

    uv run --group dev export.py --ckpt ../runs/reader-delta3/best_acc --out conikud.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import onnx
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../runs/reader-delta3/best_acc")
    p.add_argument("--out", default="conikud.onnx")
    return p.parse_args()


class ExportModel(nn.Module):
    """GraphReader.forward with pure-tensor inputs (no python span lists).
    One sentence per call: cond is [1, 2], b maps each word to row 0."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, s, e, letter_ids, lens, cond):
        m = self.m
        # FiLM is zeroed in this lineage — all conditioning lives in the delta
        # head. Skipping it at trace time drops those nodes from the graph.
        m._film_state["ce"] = None
        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        b = torch.zeros_like(s)
        # fixed window (masked): traceable, no dynamic arange
        wt = torch.arange(32, device=h.device)
        mem = m.proj(h[b.unsqueeze(1), (s.unsqueeze(1) + wt).clamp(max=h.shape[1] - 1)])
        mem_pad = wt.unsqueeze(0) >= (e - s).unsqueeze(1)
        L = letter_ids.shape[1]
        lpos = torch.arange(L, device=h.device)
        q = m.letter_emb(letter_ids) + m.pos_emb(lpos).unsqueeze(0)
        let_pad = torch.arange(L, device=h.device).unsqueeze(0) >= lens.unsqueeze(1)
        x = m.decoder(q, mem, tgt_key_padding_mask=let_pad, memory_key_padding_mask=mem_pad)
        out = m.head(x)
        wc = cond[b]
        ce = torch.cat([m.delta_emb_s(wc[:, 0]), m.delta_emb_l(wc[:, 1])], -1)
        delta = m.delta_head(torch.cat([x, ce.unsqueeze(1).expand(-1, L, -1)], -1))
        out = out + delta * (wc.sum(-1) > 0).float()[:, None, None]
        from model import NUM_C, NUM_V
        return out[..., :NUM_C], out[..., NUM_C:NUM_C + NUM_V], out[..., NUM_C + NUM_V:]


def main():
    args = parse_args()
    from graph import FINAL_HET, OPTIONS
    from model import MAX_LETTERS, load, tokenizer

    m = load(args.ckpt, "cpu")
    m.eval()
    ex = ExportModel(m)

    tok = tokenizer()
    enc = tok(["שלום עולם"], return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    s = torch.tensor([1, 3])
    e = torch.tensor([3, 5])
    lids = torch.zeros(2, MAX_LETTERS, dtype=torch.long)
    lens = torch.tensor([4, 4])
    cond = torch.zeros(1, 2, dtype=torch.long)

    torch.onnx.export(
        ex, (ids, am, s, e, lids, lens, cond), args.out,
        input_names=["input_ids", "attention_mask", "span_start", "span_end",
                     "letter_ids", "lens", "cond"],
        output_names=["cons", "vow", "stress"],
        dynamic_axes={
            "input_ids": {1: "tokens"}, "attention_mask": {1: "tokens"},
            "span_start": {0: "words"}, "span_end": {0: "words"},
            "letter_ids": {0: "words"}, "lens": {0: "words"},
            "cons": {0: "words"}, "vow": {0: "words"}, "stress": {0: "words"},
        },
        opset_version=17,
        dynamo=False,  # legacy tracer: handles the FiLM shared-state dict
    )

    onx = onnx.load(args.out)
    def put(k, v):
        p = onx.metadata_props.add()
        p.key, p.value = k, v
    put("tokenizer", tok.backend_tokenizer.to_str())
    put("options", json.dumps(
        {k: [[o.cons, o.vowel, o.plain, o.stressed] for o in v] for k, v in OPTIONS.items()},
        ensure_ascii=False))
    put("final_het", json.dumps(
        [[o.cons, o.vowel, o.plain, o.stressed] for o in FINAL_HET], ensure_ascii=False))
    put("max_letters", str(MAX_LETTERS))
    onnx.save(onx, args.out)
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    # int8 dynamic quantization (weights only) — ~4x smaller, faster on CPU
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out8 = str(Path(args.out).with_stem(Path(args.out).stem + "_int8"))
    skip = [n.name for n in onx.graph.node
            if "cond_proj" in n.name or "delta_" in n.name or "cond_emb" in n.name]
    quantize_dynamic(args.out, out8, weight_type=QuantType.QInt8, nodes_to_exclude=skip)
    onx8 = onnx.load(out8)
    if not any(p.key == "tokenizer" for p in onx8.metadata_props):
        for p in onx.metadata_props:
            q = onx8.metadata_props.add()
            q.key, q.value = p.key, p.value
        onnx.save(onx8, out8)
    print(f"quantized {out8} ({Path(out8).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
