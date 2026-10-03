"""
Cross-modal consistency checks. Each finding is:
  {"pair": "image-text", "score": 0-1 (1 = consistent), "finding": str, "raw": {...}}
Models load lazily on first use and are cached. A failing check is skipped, never fatal.
"""
import logging
import re
from functools import lru_cache
import math

log = logging.getLogger(__name__)

CLIP_MODEL = "openai/clip-vit-base-patch32"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# Score a claim relative to unrelated generic captions for the same image.
# Starting values: tune on the demo cases in Step 6.
BANK_MARGIN = 0.02   # how far above the baseline counts as a 50/50 match
BANK_SLOPE = 60.0    # how sharply the score rises with that gap

BANK = [
    "a landscape with trees", "a city street with traffic", "food on a table",
    "an animal outdoors", "a tall building", "people standing in a room",
    "a car on a road", "a printed document", "a crowd of people",
    "a river or lake", "a beach", "a forest", "a computer screen",
    "a screenshot of a website", "a close-up of a face", "a house",
    "the sky and clouds", "a drawing or illustration", "a mountain",
    "an empty room",
]


def consistency_from_delta(delta: float) -> float:
    """Map (claim similarity - baseline similarity) to a 0..1 consistency score."""
    return float(1.0 / (1.0 + math.exp(-BANK_SLOPE * (delta - BANK_MARGIN))))

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


# ---------------- Image <-> claim (CLIP) ----------------

@lru_cache(maxsize=1)
def _clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        return model, processor
    except Exception as exc:
        log.warning("CLIP unavailable: %s", exc)
        return None


def image_text_finding(image_path: str, claims: list):
    loaded = _clip()
    if loaded is None or not claims:
        return None
    import torch
    from PIL import Image

    model, processor = loaded
    image = Image.open(image_path).convert("RGB")
    texts = list(claims) + BANK
    inputs = processor(
        text=texts, images=image, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    )
    with torch.no_grad():
        out = model(**inputs)
    sims = (out.image_embeds @ out.text_embeds.T).squeeze(0)  # cosine similarities

    claim_sim = float(sims[: len(claims)].mean())
    bank_mean = float(sims[len(claims):].mean())
    delta = claim_sim - bank_mean
    score = consistency_from_delta(delta)

    if score >= 0.7:
        text = "Image content is consistent with the claim"
    elif score <= 0.4:
        text = "Image content does not match the claim"
    else:
        text = "Image only weakly matches the claim"
    return {"pair": "image-text", "score": round(score, 2), "finding": text,
            "raw": {"claim_similarity": round(claim_sim, 3),
                    "baseline": round(bank_mean, 3), "delta": round(delta, 3)}} 


# ---------------- Transcript <-> claim (NLI) ----------------

@lru_cache(maxsize=1)
def _nli():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        labels = [str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)]
        if labels[0].startswith("label"):  # generic names -> assume the standard order
            labels = ["contradiction", "entailment", "neutral"]
        return tokenizer, model, labels
    except Exception as exc:
        log.warning("NLI model unavailable: %s", exc)
        return None


def transcript_claim_finding(transcript: str, claims: list):
    loaded = _nli()
    if loaded is None or not transcript or not claims:
        return None
    import torch

    tokenizer, model, labels = loaded
    scores, contradictions = [], []
    for claim in claims:
        enc = tokenizer(transcript, claim, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = dict(zip(labels, torch.softmax(logits, dim=-1).tolist()))
        scores.append(probs["entailment"] + 0.5 * probs["neutral"])
        contradictions.append(probs["contradiction"])

    score = _clamp01(sum(scores) / len(scores))
    contradiction = max(contradictions)

    if contradiction >= 0.5:
        text = "Transcript contradicts the claim"
    elif score >= 0.7:
        text = "Transcript supports the claim"
    else:
        text = "Transcript does not clearly support the claim"
    return {"pair": "audio-text", "score": round(score, 2), "finding": text,
            "raw": {"contradiction": round(contradiction, 3)}}


# ---------------- Metadata <-> claim (rules) ----------------

def metadata_text_finding(metadata, claim_text: str):
    taken = (metadata or {}).get("datetime")
    if not taken or not claim_text:
        return None
    exif_years = YEAR_RE.findall(str(taken))
    claim_years = YEAR_RE.findall(claim_text)
    if not exif_years or not claim_years:
        return None

    exif_year = exif_years[0]
    if exif_year in claim_years:
        return {"pair": "metadata-text", "score": 0.9,
                "finding": f"Photo timestamp ({exif_year}) matches the year in the claim"}
    return {"pair": "metadata-text", "score": 0.1,
            "finding": f"Photo timestamp ({exif_year}) does not match the year in the claim ({claim_years[0]})"}


# ---------------- Entry point ----------------

def _safe(label: str, func, *args):
    try:
        return func(*args)
    except Exception as exc:
        log.warning("%s check failed: %s", label, exc)
        return None


def run_crossmodal(case: dict, modality_results: dict, text_result: dict | None) -> list:
    claims = (text_result or {}).get("claims") or []
    if not claims:
        return []  # nothing to compare against

    findings = []

    if case.get("image"):
        finding = _safe("image-text", image_text_finding, case["image"], claims)
        if finding:
            findings.append(finding)

        audio = modality_results.get("audio") or {}
    transcript = (
        audio.get("transcript")
        or (audio.get("transcription") or {}).get("text")
        or ""
    ).strip()
    if transcript:
        finding = _safe("audio-text", transcript_claim_finding, transcript, claims)
        if finding:
            findings.append(finding)

    image_meta = (modality_results.get("image") or {}).get("metadata")
    finding = _safe("metadata-text", metadata_text_finding, image_meta, " ".join(claims))
    if finding:
        findings.append(finding)

    return findings