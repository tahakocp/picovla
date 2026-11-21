"""Instruction templates and a word tokenizer (no pretrained text model)."""
import re

TRAIN = {
    "touch": ["touch the {c} block", "reach the {c} block", "tap {c}", "go to the {c} one"],
    "push": ["push the {c} block {d}", "move the {c} block {d}", "slide {c} {d}", "push {c} to the {d}"],
    "toward": ["push the {c} block to the {o} block", "move {c} next to {o}", "bring the {c} one to the {o} one"],
}
# Never seen in training. Used to measure how far the word model generalises.
HELDOUT = {
    "touch": ["poke the {c} block", "put your finger on {c}"],
    "push": ["shove the {c} block {d}", "{c} block goes {d}"],
    "toward": ["slide the {c} block over to {o}", "{c} to {o}"],
}
WORDS = sorted({w for k in TRAIN.values() for s in k for w in re.findall(r"[a-z]+", s.format(c="", d="", o=""))}
               | {"red", "green", "blue", "yellow", "left", "right", "up", "down"})
VOCAB = ["<pad>", "<unk>"] + WORDS
MAX_LEN = 10


def phrase(task, rng, split="train"):
    pool = (TRAIN if split == "train" else HELDOUT)[task.kind]
    s = pool[rng.integers(len(pool))]
    return s.format(c=task.color, d=task.arg, o=task.arg)


def tokenize(text):
    ids = [VOCAB.index(w) if w in VOCAB else 1 for w in re.findall(r"[a-z]+", text.lower())][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))
