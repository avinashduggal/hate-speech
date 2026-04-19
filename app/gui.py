"""
Hate Speech Classifier - Gradio GUI.

Launch:
    cd "/Users/avinashduggal/CS 273/hate-speech"
    python app/gui.py

Requires: gradio, nltk, torch, transformers
"""

import os
import random
import sys

# Transformers auto-detects TF if installed; importing TF into this process on macOS
# causes a libc++ mutex-corruption crash with torch+gradio loaded alongside.
# Forcing USE_TF=0 tells transformers to skip TF entirely. Must be set before the
# `import transformers` below.
os.environ.setdefault("USE_TF", "0")

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO)

# CPU by default; MPS has known crashes with DeBERTa disentangled attention + output_attentions=True.
# Set HATE_SPEECH_DEVICE=mps or =cuda to override.
_device_override = os.environ.get("HATE_SPEECH_DEVICE", "").lower()
if _device_override in ("mps", "cuda", "cpu"):
    DEVICE = _device_override
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

MODEL_CONFIGS = {
    "DeBERTa (clean)":       ("deberta", os.path.join(REPO, "model_training", "deberta-v3-hs-tuned")),
    "DeBERTa (adversarial)": ("deberta", os.path.join(REPO, "model_training", "deberta-v3-adversarial-final")),
    "RoBERTa":               ("roberta", os.path.join(REPO, "model_training", "final_roberta_model")),
}
LABELS = {0: "Non-hateful", 1: "Hateful"}

HATE_WORDS = {
    "hate", "stupid", "idiot", "trash", "kill", "dumb", "ugly", "loser",
    "worthless", "pathetic", "scum", "disgusting", "moron",
}

LEET_MAP = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})

MODEL_CACHE: dict = {}
_WORDNET_OK = False


def _ensure_wordnet() -> bool:
    global _WORDNET_OK
    if _WORDNET_OK:
        return True
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet  # noqa: F401
        _WORDNET_OK = True
    except Exception as e:
        print(f"[warn] WordNet unavailable: {e}")
        _WORDNET_OK = False
    return _WORDNET_OK


def load_model(label: str):
    if label in MODEL_CACHE:
        return MODEL_CACHE[label]
    kind, path = MODEL_CONFIGS[label]
    if kind == "deberta":
        tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
        mdl = DebertaV2ForSequenceClassification.from_pretrained(path)
    else:
        tok = AutoTokenizer.from_pretrained(path)
        mdl = RobertaForSequenceClassification.from_pretrained(path)
    mdl.to(DEVICE).eval()
    MODEL_CACHE[label] = (tok, mdl)
    return tok, mdl


def _tokenize(tokenizer, text: str):
    return tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")


def _forward(text: str, model_label: str, want_attentions: bool):
    tok, mdl = load_model(model_label)
    enc = _tokenize(tok, text)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc, output_attentions=want_attentions)
    probs = F.softmax(out.logits, dim=-1)[0].cpu().tolist()
    pred = int(torch.argmax(out.logits, dim=-1).item())
    return tok, enc, out, probs, pred


def classify(text: str, model_label: str):
    if not text or not text.strip():
        return {}, []
    _, enc, out, probs, pred = _forward(text, model_label, want_attentions=True)
    probs_dict = {LABELS[i]: float(probs[i]) for i in range(len(probs))}
    heatmap = _heatmap_from(enc, out, model_label)
    return probs_dict, heatmap


def _heatmap_from(enc, out, model_label: str):
    tok, _ = MODEL_CACHE[model_label]
    if getattr(out, "attentions", None) is None or len(out.attentions) == 0:
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        scores = [0.0] * len(tokens)
    else:
        att = out.attentions[-1][0].mean(0)          # [seq, seq]
        cls_row = att[0].cpu().tolist()              # CLS → each token
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        scores = cls_row
    return _group_tokens_to_words(tokens, scores)


def _group_tokens_to_words(tokens, scores):
    skip = {"[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"}
    words, word_scores = [], []
    cur, cur_scores = "", []

    def flush():
        if cur:
            words.append(cur)
            word_scores.append(max(cur_scores) if cur_scores else 0.0)

    for tok, s in zip(tokens, scores):
        if tok in skip:
            continue
        is_new = tok.startswith("▁") or tok.startswith("Ġ")
        clean = tok.lstrip("▁Ġ")
        if is_new and cur:
            flush()
            cur, cur_scores = "", []
        cur += clean
        cur_scores.append(float(s))
    flush()

    if not words:
        return []
    m = max(word_scores) or 1.0
    return [(w + " ", v / m) for w, v in zip(words, word_scores)]


def apply_leet(text: str) -> str:
    tokens = text.split()
    out, changed = [], 0
    for t in tokens:
        low = "".join(ch for ch in t.lower() if ch.isalpha())
        target = low in HATE_WORDS or (t.isalpha() and random.random() < 0.3)
        if target and changed < 4:
            out.append(t.translate(LEET_MAP))
            changed += 1
        else:
            out.append(t)
    return " ".join(out)


def apply_synonym(text: str) -> str:
    if not _ensure_wordnet():
        return text
    from nltk.corpus import wordnet as wn

    tokens = text.split()
    candidates = [i for i, t in enumerate(tokens) if t.isalpha() and len(t) >= 4]
    if not candidates:
        return text
    random.shuffle(candidates)
    swaps = 0
    for i in candidates:
        if swaps >= 2:
            break
        word = tokens[i]
        syns = wn.synsets(word)
        if not syns:
            continue
        lemma = next(
            (l.name().replace("_", " ") for s in syns for l in s.lemmas() if l.name().lower() != word.lower()),
            None,
        )
        if lemma:
            tokens[i] = lemma
            swaps += 1
    return " ".join(tokens)


def adversarial(text: str, model_label: str, mode: str):
    if not text or not text.strip():
        return "", {}, "Enter some text first."
    random.seed()  # non-deterministic for demo variety
    perturbed = text
    if mode in ("leet", "both"):
        perturbed = apply_leet(perturbed)
    if mode in ("synonym", "both"):
        perturbed = apply_synonym(perturbed)

    orig_probs, _ = classify(text, model_label)
    new_probs, _ = classify(perturbed, model_label)

    orig_label = max(orig_probs, key=orig_probs.get) if orig_probs else "?"
    new_label = max(new_probs, key=new_probs.get) if new_probs else "?"
    flipped = orig_label != new_label
    delta = (
        f"**Original:** {orig_label} ({orig_probs.get(orig_label, 0):.2%})  \n"
        f"**Perturbed:** {new_label} ({new_probs.get(new_label, 0):.2%})  \n"
        f"**Attack {'succeeded (label flipped!)' if flipped else 'failed (label held).'}**"
    )
    return perturbed, new_probs, delta


with gr.Blocks(title="Hate Speech Classifier") as demo:
    gr.Markdown("# Hate Speech Classifier\nBinary hateful / non-hateful classifier with word-importance heatmap and adversarial attack demo.")

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value="DeBERTa (clean)",
            label="Model",
        )

    text_in = gr.Textbox(lines=4, label="Input text", placeholder="Type or paste text here...")
    classify_btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        probs_out = gr.Label(num_top_classes=2, label="Prediction")
        heatmap_out = gr.HighlightedText(
            label="Word-importance heatmap (CLS attention, last layer)",
            combine_adjacent=False,
            show_legend=False,
        )

    gr.Markdown("---\n### Adversarial attack\nApplies leet-speak substitution and/or WordNet synonym swap to the input, then re-classifies.")

    with gr.Row():
        attack_mode = gr.Radio(
            choices=["leet", "synonym", "both"],
            value="both",
            label="Attack mode",
        )
        attack_btn = gr.Button("Attack & re-classify")

    perturbed_out = gr.Textbox(label="Perturbed text", lines=3, interactive=False)
    with gr.Row():
        new_probs_out = gr.Label(num_top_classes=2, label="Prediction on perturbed text")
        delta_out = gr.Markdown()

    classify_btn.click(classify, inputs=[text_in, model_dd], outputs=[probs_out, heatmap_out])
    attack_btn.click(adversarial, inputs=[text_in, model_dd, attack_mode], outputs=[perturbed_out, new_probs_out, delta_out])


if __name__ == "__main__":
    demo.launch()
