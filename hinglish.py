"""Devanagari -> Roman (Hinglish) transliteration.

Whisper emits Devanagari when it hears Hindi. Hinglish is Hindi written in
Roman letters the way people actually type it ("kya haal hai", "samajh"),
so we convert any Devanagari that comes out of the model. Latin text
(English words the model already wrote in Roman) passes through untouched.
"""
import re
import unicodedata

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
INDEP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
# matra -> (word-final form, medial form)
MATRA = {
    "ा": ("a", "aa"), "ि": ("i", "i"), "ी": ("i", "ee"), "ु": ("u", "u"),
    "ू": ("u", "oo"), "ृ": ("ri", "ri"), "े": ("e", "e"), "ै": ("ai", "ai"),
    "ो": ("o", "o"), "ौ": ("au", "au"), "ॉ": ("o", "o"), "ॅ": ("e", "e"),
}
DIGITS = {ord("०") + i: str(i) for i in range(10)}
DEV_WORD = re.compile(r"[ऀ-ॿ]+")
HAS_DEV = re.compile(r"[ऀ-ॿ]")


def _word(w: str) -> str:
    w = unicodedata.normalize("NFD", w)
    # units: [roman_consonant, vowel_or_None, is_inherent]
    units = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        if c in INDEP:
            units.append(["", INDEP[c], False])
            i += 1
        elif c in CONS:
            if c == "क" and w[i + 1:i + 3] == "्ष":  # क्ष
                cons, i = "ksh", i + 3
            elif c == "ज" and w[i + 1:i + 3] == "्ञ":  # ज्ञ
                cons, i = "gy", i + 3
            else:
                cons, i = CONS[c], i + 1
                if i < n and w[i] == "़":
                    cons = NUKTA.get(c, cons)
                    i += 1
            if i < n and w[i] == "्":  # halant: bare consonant
                units.append([cons, None, False])
                i += 1
            elif i < n and w[i] in MATRA:
                units.append([cons, w[i], False])
                i += 1
            else:
                units.append([cons, "a", True])
        elif c in "ंँ":
            units.append(["n", None, False])
            i += 1
        elif c == "ः":
            units.append(["h", None, False])
            i += 1
        else:
            units.append([c, None, False])
            i += 1

    # schwa deletion, right to left
    last = len(units) - 1
    for k in range(last, -1, -1):
        u = units[k]
        if not u[2]:
            continue
        if k == last:
            u[1] = None
        elif 0 < k and units[k - 1][1] and units[k + 1][1]:
            u[1] = None

    out = []
    for k, (cons, v, inherent) in enumerate(units):
        if v is None:
            out.append(cons)
        elif v in MATRA:
            # a matra is "final" when nothing but anusvara/visarga follows
            tail = units[k + 1:]
            final = all(t[0] in ("n", "h") and t[1] is None for t in tail)
            out.append(cons + MATRA[v][0 if final else 1])
        else:
            out.append(cons + v)
    return "".join(out)


def to_hinglish(text: str) -> str:
    if not HAS_DEV.search(text):
        return text
    text = text.translate(DIGITS).replace("।", ".").replace("॥", ".")
    return DEV_WORD.sub(lambda m: _word(m.group(0)), text)
