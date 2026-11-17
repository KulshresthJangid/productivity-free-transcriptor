"""Drop segments that are not real speech (laughter, music, Whisper hallucinations)."""
import re

# Whisper invents these on silence/noise/music.
HALLUCINATIONS = [
    "thanks for watching", "thank you for watching", "please subscribe", "subscribe to my channel",
    "like and subscribe", "see you in the next video", "subtitles by", "amara.org",
    "transcribed by", "www.", ".com",
    "देखने के लिए धन्यवाद", "सब्सक्राइब", "धन्यवाद",
    "dekhne ke liye dhanyavaad", "subscribe karna", "subscribe kar",
]
# [Music], (laughs), ♪ ... , *applause*
TAG = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]|\*[^*]{0,40}\*|[♪♫]+")
# haha, hahaha, hehe, hmm, mmm, uhh, ummm, ahh
NOISE_WORD = re.compile(
    r"\b(?:(?:ha|he|hi|ho){2,}|h?m{2,}|um+|uh+)\b[,.!?]*",
    re.I)
WORD = re.compile(r"\w+", re.UNICODE)


def clean_text(text: str) -> str:
    text = TAG.sub(" ", text)
    text = NOISE_WORD.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,.-")


def is_junk(seg, text: str) -> bool:
    """seg is a faster-whisper Segment; text is already cleaned."""
    if not WORD.search(text):
        return True
    low = text.lower()
    if any(h in low for h in HALLUCINATIONS) and len(WORD.findall(text)) <= 8:
        return True
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
        return True            # model itself thinks this is silence/noise
    if seg.avg_logprob < -1.5:
        return True            # gibberish: very low confidence
    if seg.compression_ratio > 2.4:
        return True            # repetitive loop ("the the the the")
    return False


class RepeatFilter:
    """Whisper sometimes repeats the same line over and over; keep the first few."""

    def __init__(self, limit=2):
        self.limit, self.last, self.count = limit, "", 0

    def keep(self, text: str) -> bool:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key == self.last:
            self.count += 1
        else:
            self.last, self.count = key, 1
        return self.count <= self.limit
